"""Ixchel (la tejedora): mezcla de cadenas de expertos de CONTEO con suavizado Kneser-Ney modificado
(descuentos en forma cerrada) y pesos de mezcla por EM con compuertas discretas (sin learning rate,
sin gradientes). Todo es conteo + algebra cerrada + iteracion de punto fijo (EM).

Arquitectura (v0.1):
  - ATOMOS: rasgos discretos del pasado (char t-k, clase de char, prefijo de palabra, palabra previa,
    posicion en linea, ...), calculados en una pasada (numba).
  - CADENA: lista de NIVELES de contexto de grueso a fino; cada nivel = tupla de atomos. Cada cadena es un
    modelo KN jerarquico (nivel 0 = sin contexto). El nivel mas fino usa conteos crudos; los inferiores usan
    conteos de continuacion (KN). Descuentos D1,D2,D3+ cerrados a partir de los count-of-counts.
  - MEZCLA: p(x) = sum_k w[g, k] p_k(x); g = compuerta discreta (nivel mas largo visto y su conteo por cadena).
    w se estima por EM sobre un tramo dev interno del TRAIN (nunca val); luego se recuentan las tablas con todo el train.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
import numba as nb
import json, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixing import poe_global, poe_combine

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # PATCH r3: copia en experiments/r3_engine (2 niveles bajo la raiz)
DATA = os.path.join(ROOT, "data", os.environ.get("IXCHEL_DATA", "shakespeare"))

# ---------------------------------------------------------------- atomos
ATOM_NAMES = {}
for k in range(1, 13):
    ATOM_NAMES[f"c{k}"] = k - 1            # 0..11 caracter t-k
for k in range(1, 9):
    ATOM_NAMES[f"k{k}"] = 11 + k           # 12..19 clase de caracter t-k
ATOM_NAMES.update(dict(wp=20, pw1=21, pw2=22, lpos=23, wlen=24, lfirst=25, pw1f=26, sp=27))
NATOM = 28
# wp: hash del prefijo de la palabra actual; pw1/pw2: hash de las 2 palabras previas completas;
# lpos: chars desde el ultimo salto de linea (tope 63); wlen: largo del prefijo actual (tope 15);
# lfirst: primer caracter de la linea actual (255 si ninguno); pw1f: primer char de la palabra previa;
# sp: 1 si hay un salto de linea doble reciente ("estamos al inicio de un bloque")


def build_char_tables(itos):
    V = len(itos)
    cls = np.zeros(256, dtype=np.uint8)   # 0 = pad
    isletter = np.zeros(256, dtype=np.uint8)
    isnl = np.zeros(256, dtype=np.uint8)
    for i, ch in enumerate(itos):
        if ch in "aeiouAEIOU":
            cls[i] = 1 if ch.islower() else 3
            isletter[i] = 1
        elif ch.isalpha():
            cls[i] = 2 if ch.islower() else 4
            isletter[i] = 1
        elif ch == " ":
            cls[i] = 5
        elif ch == "\n":
            cls[i] = 6
            isnl[i] = 1
        elif ch in ".,;:!?":
            cls[i] = 7
        else:
            cls[i] = 8   # digitos, comillas, apostrofe, guiones, corchetes...
    return cls, isletter, isnl


@nb.njit(cache=True)
def _atoms(ids, cls, isletter, isnl, cols, is64, dst, out8, out64):
    n = ids.shape[0]
    nc = cols.shape[0]
    a = np.zeros(28, dtype=np.int64)
    wp = np.uint64(0)
    pw1 = np.uint64(0)
    pw2 = np.uint64(0)
    wlen = 0
    lpos = 0
    lfirst = 255
    pw1f = 255
    wfirst = 255
    sp = 0
    P = np.uint64(1099511628211)
    for t in range(n):
        for k in range(12):
            c = ids[t - k - 1] if t - k - 1 >= 0 else 255
            a[k] = c
            if k < 8:
                a[12 + k] = cls[c] if c < 255 else 0
        a[20] = np.int64(wp >> np.uint64(2))
        a[21] = np.int64(pw1 >> np.uint64(2))
        a[22] = np.int64(pw2 >> np.uint64(2))
        a[23] = lpos if lpos < 63 else 63
        a[24] = wlen if wlen < 15 else 15
        a[25] = lfirst
        a[26] = pw1f
        a[27] = sp
        for j in range(nc):
            if is64[j]:
                out64[t, dst[j]] = a[cols[j]]
            else:
                out8[t, dst[j]] = a[cols[j]]
        # consumir ids[t]
        c = ids[t]
        if isletter[c] == 1:
            if wlen == 0:
                wfirst = c
            wp = (wp + np.uint64(c) + np.uint64(1)) * P
            wlen += 1
        else:
            if wlen > 0:
                pw2 = pw1
                pw1 = wp
                pw1f = wfirst
            wp = np.uint64(0)
            wlen = 0
        if isnl[c] == 1:
            if lpos == 0:
                sp = 1
            else:
                sp = 0
            lpos = 0
            lfirst = 255
        else:
            if lpos == 0:
                lfirst = c
            lpos += 1


@nb.njit(inline="always")
def _mix(h):
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


@nb.njit(cache=True, parallel=True)
def _level_hashes(A8, A64, pos, colcode, lvl_off, lvl_id0, out):
    # A8 [n,n8] uint8 (atomos pequenos) ; A64 [n,n64] int64 (hashes de palabra) ; pos [m]
    # colcode: >=0 columna de A8 ; <0 columna (-c-1) de A64 ; lvl_off [L1'+1] ; lvl_id0: id global del primer nivel
    L1 = lvl_off.shape[0] - 1
    m = pos.shape[0]
    for i in nb.prange(m):
        t = pos[i]
        for l in range(L1):
            h = np.uint64(0x9E3779B97F4A7C15) * np.uint64(lvl_id0 + l + 1)
            for q in range(lvl_off[l], lvl_off[l + 1]):
                cc = colcode[q]
                if cc >= 0:
                    v = np.uint64(A8[t, cc])
                else:
                    v = np.uint64(A64[t, -cc - 1])
                h = _mix(h ^ (v + np.uint64(0x632BE59BD9B4E019)))
            out[l, i] = _mix(h)


# ---------------------------------------------------------------- tablas
MASK = np.uint64(~np.uint64(127))


def _make_table(upk, cnt):
    """upk: claves de par ordenadas (ctx<<7 | sym) ; cnt: conteos. Devuelve dict de arrays."""
    ck = upk >> np.uint64(7)
    if len(ck) == 0:
        return dict(ckey=np.zeros(0, np.uint64), cstart=np.zeros(1, np.int64), psym=np.zeros(0, np.uint8),
                    pcnt=np.zeros(0, np.int32), D=np.array([0.5, 0.75, 0.9]))
    brk = np.flatnonzero(ck[1:] != ck[:-1]) + 1
    starts = np.concatenate(([0], brk)).astype(np.int64)
    ckey = ck[starts]
    cstart = np.concatenate((starts, [len(ck)])).astype(np.int64)
    c = cnt.astype(np.int64)
    n1, n2, n3, n4 = [(c == k).sum() for k in (1, 2, 3, 4)]
    if n1 > 0 and n2 > 0 and n3 > 0:
        Y = n1 / (n1 + 2.0 * n2)
        D1 = 1 - 2 * Y * n2 / n1
        D2 = 2 - 3 * Y * n3 / n2
        D3 = 3 - 4 * Y * n4 / n3 if n4 > 0 else 2.5
        D = np.array([D1, D2, D3])
    else:
        D = np.array([0.5, 0.75, 0.9])
    D = np.clip(D, [0.05, 0.1, 0.1], [0.99, 1.99, 2.99])
    return dict(ckey=ckey, cstart=cstart, psym=(upk & np.uint64(127)).astype(np.uint8),
                pcnt=c.astype(np.int32), D=D)


def build_chain_tables(get_H, L1, x):
    """get_H(l) -> uint64 [m] hashes de contexto del nivel l (se calcula bajo demanda: memoria O(m), no O(L1*m)).
    x [m] uint8 simbolo siguiente. Devuelve lista de tablas (una por nivel): nivel < L usa conteos de
    continuacion, el nivel L (mas fino) usa conteos crudos."""
    xs = x.astype(np.uint64)
    tabs = [None] * L1
    for l in range(L1 - 1, -1, -1):
        Hl = get_H(l)
        if l == L1 - 1:
            pk = (Hl & MASK) | xs
            upk, cnt = np.unique(pk, return_counts=True)
            tabs[l] = _make_table(upk, cnt)
            child_upk, child_first = np.unique(pk, return_index=True)
            del pk
        else:
            ph = Hl[child_first]
            ppk = (ph & MASK) | (child_upk & np.uint64(127))
            uppk, ccnt = np.unique(ppk, return_counts=True)
            tabs[l] = _make_table(uppk, ccnt)
            del ppk, ph
            if l > 0:
                pk_l = (Hl & MASK) | xs
                child_upk, child_first = np.unique(pk_l, return_index=True)
                del pk_l
        del Hl
    return tabs


def _concat_tables(tabs):
    ck, cs, ps, pc = [], [], [], []
    c_off, cs_off = [0], [0]
    D = np.zeros((len(tabs), 3))
    poff = 0
    for l, t in enumerate(tabs):
        ck.append(t["ckey"])
        cs.append(t["cstart"] + poff)
        ps.append(t["psym"])
        pc.append(t["pcnt"])
        poff += len(t["psym"])
        c_off.append(c_off[-1] + len(t["ckey"]))
        cs_off.append(cs_off[-1] + len(t["cstart"]))
        D[l] = t["D"]
    return dict(ckey=np.concatenate(ck), cstart=np.concatenate(cs), psym=np.concatenate(ps),
                pcnt=np.concatenate(pc), c_off=np.array(c_off, np.int64), cs_off=np.array(cs_off, np.int64), D=D)


@nb.njit(cache=True, parallel=True)
def _chain_predict(H, ckey, cstart, psym, pcnt, c_off, cs_off, D, V, out_p, out_lhi, out_n):
    L1 = H.shape[0]
    m = H.shape[1]
    for i in nb.prange(m):
        p = np.full(V, 1.0 / V)
        lhi = -1
        nhi = 0.0
        for l in range(L1):
            key = H[l, i] >> np.uint64(7)
            lo = c_off[l]
            hi = c_off[l + 1]
            # busqueda binaria
            a = lo
            b = hi
            while a < b:
                mid = (a + b) >> 1
                if ckey[mid] < key:
                    a = mid + 1
                else:
                    b = mid
            if a >= hi or ckey[a] != key:
                break
            j = a - lo
            s = cstart[cs_off[l] + j]
            e = cstart[cs_off[l] + j + 1]
            n = 0.0
            N1 = 0.0
            N2 = 0.0
            N3 = 0.0
            for q in range(s, e):
                c = pcnt[q]
                n += c
                if c == 1:
                    N1 += 1.0
                elif c == 2:
                    N2 += 1.0
                else:
                    N3 += 1.0
            gamma = (D[l, 0] * N1 + D[l, 1] * N2 + D[l, 2] * N3) / n
            for x in range(V):
                p[x] *= gamma
            for q in range(s, e):
                c = pcnt[q]
                d = D[l, 0] if c == 1 else (D[l, 1] if c == 2 else D[l, 2])
                p[psym[q]] += (c - d) / n
            lhi = l
            nhi = n
        for x in range(V):
            out_p[i, x] = p[x]
        out_lhi[i] = lhi
        out_n[i] = nhi


def _nbucket(n):
    return np.where(n <= 1, 0, np.where(n <= 3, 1, np.where(n <= 15, 2, 3)))


# ---------------------------------------------------------------- cadenas
def default_chains():
    A = [[]] + [[f"c{k}" for k in range(1, o + 1)] for o in range(1, 9)]
    W = [[], ["wp"], ["wp", "pw1"], ["wp", "pw1", "pw2"]]
    S = [[], ["k1", "k2", "k3", "k4"], ["c1", "k2", "k3", "k4", "k5"],
         ["c1", "c2", "k3", "k4", "k5", "k6"], ["c1", "c2", "c3", "k4", "k5", "k6", "k7", "k8"]]
    Lc = [[], ["lpos"], ["lpos", "c1"], ["lfirst", "lpos", "c1", "c2"]]
    return dict(A=A, W=W, S=S, L=Lc)


from cache_experts import (CacheWB, MatchIndex, match_bucket, match_hits, match_accuracy, match_dists,
                           unigram, N_MBUCKET)


class Ixchel:
    """Ixchel v0.2 (ronda 2, codename cache) = v0.1 + expertos de memoria en contexto.

    Expertos (columnas de la mezcla, todos devuelven [m,V]):
      - cadenas KN jerarquicas (A, W, S, L) como en v0.1;
      - CACHE: un experto WB-4 por ventana de historial (cache_windows; 1<<40 = todo el historial);
      - MATCH: match model contra el historial ('hist', causal) y/o el train ('train'), con confianza por conteo.
    Compuerta (gate) = producto mixto-radix de: niveles de las cadenas de gate_chains (como v0.1) + extras
    opcionales gate_extra in {'cache_lvl' (mayor orden con conteo en la ventana cache_gate_w), 'match_b' (bucket
    de longitud del match)}. Pesos w[g,k] por EM (punto fijo, sin learning rate) sobre un tramo dev interno del
    train (nunca val). Las tablas se recuentan con todo el train antes de predecir.
    Con cache_windows=() y match_sources=() reproduce v0.1 exactamente.
    """

    def __init__(self, chains=None, gate_chains=("A", "W"), name="Ixchel-v0.2", em_iters=60, prior_m=1.0,
                 dev_frac=0.05, dev_positions=150_000, gated=True, verbose=True, t0=16,
                 cache_windows=(), match_sources=(), gate_extra=(), cache_gate_w=-1, chunk=25_000, mix="em"):
        self.mix = mix                      # 'em' (lineal con compuerta, v0.1/r2_cache) | 'poe' (geometrica global, v0.2)
        self.poe_w = None
        self.chains = chains if chains is not None else default_chains()
        self.names = list(self.chains.keys())
        self.gate_chains = [self.names.index(c) for c in gate_chains]
        self.name = name
        self.em_iters, self.prior_m, self.dev_frac, self.dev_positions = em_iters, prior_m, dev_frac, dev_positions
        self.gated, self.verbose, self.t0 = gated, verbose, t0
        self.cache_windows = tuple(cache_windows)
        self.match_sources = tuple(match_sources)
        self.gate_extra = tuple(gate_extra)
        self.cache_gate_w = cache_gate_w
        self.chunk = chunk
        self.cache = CacheWB(self.cache_windows) if self.cache_windows else None
        itos = json.load(open(os.path.join(DATA, "vocab.json")))
        self.cls, self.isletter, self.isnl = build_char_tables(itos)
        used = sorted({ATOM_NAMES[a] for ch in self.chains.values() for lvl in ch for a in lvl})
        self.cols = np.array(used, dtype=np.int64)
        self.is64 = np.array([c in (20, 21, 22) for c in used], dtype=np.bool_)       # wp, pw1, pw2 = hashes de 64 bits
        self.dst = np.zeros(len(used), dtype=np.int64)
        n8 = n64 = 0
        col_pos = {}
        for i, c in enumerate(used):
            if self.is64[i]:
                self.dst[i] = n64; col_pos[c] = -(n64 + 1); n64 += 1
            else:
                self.dst[i] = n8; col_pos[c] = n8; n8 += 1
        self.n8, self.n64 = n8, n64
        self.chain_idx = {}
        for nm, ch in self.chains.items():
            colidx = [col_pos[ATOM_NAMES[a]] for lvl in ch for a in lvl]
            off = [0]
            for lvl in ch:
                off.append(off[-1] + len(lvl))
            self.chain_idx[nm] = (np.array(colidx, np.int64), np.array(off, np.int64))
        self.tabs = {}
        self.w = None
        self.base = None
        self.hits = {}
        self.train_index = None

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _atoms_for(self, ids):
        out8 = np.zeros((len(ids), max(self.n8, 1)), dtype=np.uint8)
        out64 = np.zeros((len(ids), max(self.n64, 1)), dtype=np.int64)
        _atoms(ids, self.cls, self.isletter, self.isnl, self.cols, self.is64, self.dst, out8, out64)
        return (out8, out64)

    def _H(self, A, pos, nm):
        colidx, off = self.chain_idx[nm]
        H = np.zeros((len(off) - 1, len(pos)), dtype=np.uint64)
        _level_hashes(A[0], A[1], pos, colidx, off, 0, H)
        return H

    def _H_level(self, A, pos, nm, l):
        """Hash de UN nivel de la cadena nm (memoria O(m))."""
        colidx, off = self.chain_idx[nm]
        sub = np.array([0, off[l + 1] - off[l]], dtype=np.int64)
        H = np.zeros((1, len(pos)), dtype=np.uint64)
        _level_hashes(A[0], A[1], pos, colidx[off[l]:off[l + 1]] if off[l + 1] > off[l] else colidx[:0], sub, l, H)
        return H[0]

    def _count(self, A, x, lo, hi):
        pos = np.arange(lo, hi, dtype=np.int64)
        xs = x[lo:hi]
        tabs = {}
        for nm in self.names:
            L1 = len(self.chain_idx[nm][1]) - 1
            tabs[nm] = _concat_tables(build_chain_tables(lambda l, nm=nm: self._H_level(A, pos, nm, l), L1, xs))
        return tabs

    def _chain_dists(self, tabs, A, pos, V):
        P, LH, NN = [], [], []
        for nm in self.names:
            H = self._H(A, pos, nm)
            T = tabs[nm]
            p = np.zeros((len(pos), V), dtype=np.float64)
            lhi = np.zeros(len(pos), dtype=np.int64)
            nn = np.zeros(len(pos), dtype=np.float64)
            _chain_predict(H, T["ckey"], T["cstart"], T["psym"], T["pcnt"], T["c_off"], T["cs_off"], T["D"], V,
                           p, lhi, nn)
            P.append(p); LH.append(lhi); NN.append(nn)
        return P, LH, NN

    def _chain_feats(self, LH, NN):
        """Caracteristicas de compuerta de las cadenas de gate_chains (identicas a v0.1)."""
        feats = []
        for ci in self.gate_chains:
            L1 = len(self.chains[self.names[ci]])
            feats.append(((LH[ci] + 1) * 4 + _nbucket(NN[ci]), (L1 + 1) * 4))
        return feats

    @staticmethod
    def _combine(feats):
        """Mixto-radix de las caracteristicas (arr, cardinalidad) -> (indice de compuerta [m], G)."""
        if not feats:
            return None, 1
        n = len(feats[0][0])
        g = np.zeros(n, dtype=np.int64)
        G = 1
        for arr, card in feats:
            g = g * card + arr
            G *= card
        return g, G

    def _ctx_prepare(self, ids, maxp, base, train_idx):
        """Estado de los expertos en contexto construido desde ids[:maxp] (historial) y el train."""
        ctx = dict(base=base)
        if self.cache is not None:
            ctx["cache_idx"] = self.cache.index(ids, maxp)
        if self.match_sources:
            ctx["match"] = {}
            for s in self.match_sources:
                ctx["match"][s] = MatchIndex(ids, maxp, causal=True) if s == "hist" else train_idx
        return ctx

    def _expert_dists(self, tabs, A, chain_pos, ids, pos, V, ctx):
        """Lista de P [m,V] de TODOS los expertos + lista de caracteristicas de compuerta.
        chain_pos: posiciones sobre A (atomos del train completo en dev). pos: posiciones sobre ids (historial)."""
        P, LH, NN = self._chain_dists(tabs, A, chain_pos, V)
        Ps = list(P)
        feats = self._chain_feats(LH, NN) if self.gated else []
        if self.cache is not None:
            Pc, lvl = self.cache.query(ctx["cache_idx"], pos, ctx["base"])
            Ps += [Pc[:, w, :] for w in range(self.cache.nw)]
            if "cache_lvl" in self.gate_extra:
                feats.append((lvl[:, self.cache_gate_w], 5))
        if self.match_sources:
            mb = np.zeros(len(pos), dtype=np.int64)
            for s in self.match_sources:
                sym, L = ctx["match"][s].raw(ids, pos)
                Ps.append(match_dists(sym, L, ctx["base"], self.hits[s]))
                mb = np.maximum(mb, match_bucket(L))
            if "match_b" in self.gate_extra:
                feats.append((mb, N_MBUCKET))
        return Ps, feats

    def expert_names(self):
        names = list(self.names)
        if self.cache is not None:
            names += [f"C{int(w) if w < (1 << 40) else 'inf'}" for w in self.cache_windows]
        names += [f"M_{s}" for s in self.match_sources]
        return names

    def fit(self, train, V):
        self.V = V
        t_start = time.time()
        n = len(train)
        A = self._atoms_for(train)
        cut = int(n * (1 - self.dev_frac))
        self.log(f"[{self.name}] contando sobre train[:{cut}] ...")
        tabs = self._count(A, train, self.t0, cut)
        # ---- dev interno: cadenas con tablas de train[:cut]; expertos en contexto con historial = train[cut:]
        dev_pos = np.linspace(cut, n - 1, min(self.dev_positions, n - cut)).astype(np.int64)
        q = dev_pos - cut
        ids_dev = train[cut:]
        base_dev = unigram(train[:cut], V)
        dev_train_idx = MatchIndex(train[:cut], cut, causal=False) if "train" in self.match_sources else None
        ctx = self._ctx_prepare(ids_dev, int(q.max()), base_dev, dev_train_idx)
        xd = train[dev_pos]
        if self.match_sources:
            # calibracion de la confianza del match SOLO con dev (conteo de aciertos por bucket)
            for s in self.match_sources:
                sym, L = ctx["match"][s].raw(ids_dev, q)
                hit, tot = match_hits(sym, L, xd)
                self.hits[s] = match_accuracy(hit, tot, self.prior_m)
        ptrue_rows, gate_rows, G = [], [], 1
        LPdev = None
        for c0 in range(0, len(dev_pos), self.chunk):
            sl = slice(c0, min(c0 + self.chunk, len(dev_pos)))
            Ps, feats = self._expert_dists(tabs, A, dev_pos[sl], ids_dev, q[sl], V, ctx)
            if self.mix == "poe":
                if LPdev is None:
                    LPdev = np.zeros((len(Ps), len(dev_pos), V), dtype=np.float32)
                for k, p in enumerate(Ps):
                    LPdev[k, sl] = np.log(np.maximum(p, 1e-12))
            ptrue_rows.append(np.stack([p[np.arange(len(p)), xd[sl]] for p in Ps], axis=1))
            g, G = self._combine(feats)
            gate_rows.append(g if g is not None else np.zeros(sl.stop - sl.start, np.int64))
        ptrue = np.concatenate(ptrue_rows)
        gate = np.concatenate(gate_rows)
        self.G = G
        names = self.expert_names()
        self.dev_bpc_experts = {nm: float(-np.log2(np.maximum(ptrue[:, k], 1e-12)).mean())
                                for k, nm in enumerate(names)}
        self.dev_bpc_chains = {nm: self.dev_bpc_experts[nm] for nm in self.names}
        self.w = self._em(ptrue, gate, G)
        mix = (self.w[gate] * ptrue).sum(1)
        self.dev_bpc_mix = float(-np.log2(np.maximum(mix, 1e-12)).mean())
        if self.mix == "poe":
            self.poe_w = poe_global(LPdev, xd, verbose=self.verbose)
            L = np.zeros((len(dev_pos), V))
            for k in range(LPdev.shape[0]):
                L += self.poe_w[k] * LPdev[k]
            L -= L.max(1, keepdims=True)
            L -= np.log(np.exp(L).sum(1, keepdims=True))
            self.dev_bpc_mix_em = self.dev_bpc_mix
            if getattr(self, 'keep_dev', False):
                self.dev_logp = L.astype(np.float32); self.dev_pos_used = dev_pos.copy()
            self.dev_bpc_mix = float(-L[np.arange(len(dev_pos)), xd].mean() / np.log(2))
            self.log(f"[{self.name}] PoE w={ {nm: round(float(w), 3) for nm, w in zip(names, self.poe_w)} }  dev bpc PoE={self.dev_bpc_mix:.4f} (EM={self.dev_bpc_mix_em:.4f})")
            del LPdev
        self.log(f"[{self.name}] dev bpc expertos: { {k: round(v, 3) for k, v in self.dev_bpc_experts.items()} }"
                 f"  mezcla: {self.dev_bpc_mix:.4f}  hits: { {s: np.round(h, 3).tolist() for s, h in self.hits.items()} }"
                 f"  (t={time.time()-t_start:.1f}s)")
        del ptrue_rows, ctx
        # ---- recuento final con todo el train (cadenas + indice de match sobre train)
        self.base = unigram(train, V)
        self.tabs = self._count(A, train, self.t0, n)
        if "train" in self.match_sources:
            self.train_index = MatchIndex(train, n, causal=False)
        self.fit_time = time.time() - t_start
        self.log(f"[{self.name}] listo en {self.fit_time:.1f}s ; params(celdas)={self.n_params():,}")

    def _em(self, ptrue, gate, G):
        K = ptrue.shape[1]
        w = np.full((G, K), 1.0 / K)
        wg = np.full(K, 1.0 / K)
        cnt = np.bincount(gate, minlength=G).astype(np.float64)
        for it in range(self.em_iters):
            num = w[gate] * ptrue
            r = num / np.maximum(num.sum(1, keepdims=True), 1e-300)
            wg = r.mean(0)
            acc = np.stack([np.bincount(gate, weights=r[:, k], minlength=G) for k in range(K)], axis=1)
            w = (acc + self.prior_m * wg[None, :]) / (cnt[:, None] + self.prior_m)
        return w

    def n_params(self):
        """Celdas de tablas aprendidas: pares de las cadenas + 7 tasas de acierto por fuente de match.
        (El cache no tiene parametros; su estado es el propio historial. Los pesos de mezcla no se cuentan, como en v0.1.)"""
        n = int(sum(len(t["psym"]) for tb in self.tabs.values() for t in [tb]))
        return n + N_MBUCKET * len(self.hits)

    def predict(self, ids, pos):
        pos = np.asarray(pos, dtype=np.int64)
        A = self._atoms_for(ids)
        maxp = int(pos.max()) if len(pos) else 0
        ctx = self._ctx_prepare(ids, maxp, self.base, self.train_index)
        Ps, feats = self._expert_dists(self.tabs, A, pos, ids, pos, self.V, ctx)
        if self.mix == "poe":
            return poe_combine([np.log(np.maximum(p, 1e-12)) for p in Ps], self.poe_w)
        g, _ = self._combine(feats)
        if g is None:
            g = np.zeros(len(pos), np.int64)
        W = self.w[g]
        out = np.zeros((len(pos), self.V))
        for k in range(len(Ps)):
            out += W[:, k:k + 1] * Ps[k]
        out = np.maximum(out, 1e-9)
        return (out / out.sum(1, keepdims=True)).astype(np.float32)

"""Ixchel (la tejedora) - copia de trabajo de la ronda 2 (codename lexstruct).
Base: ixchel/core.py v0.1 (KN jerarquico por cadenas + mezcla EM con compuerta discreta, sin learning rate).
Cambios lexstruct: atomos nuevos (hablante, posicion en el discurso, verso: silabas y final de linea previa,
estado de mayusculas de la palabra) y cadenas nuevas en lex_chains(). Las tablas se recuentan igual que en v0.1.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
import numba as nb
import json, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # experiments/r2_lexstruct -> arquitectura-nueva
DATA = os.path.join(ROOT, "data", "shakespeare")

# ---------------------------------------------------------------- atomos
ATOM_NAMES = {}
for k in range(1, 13):
    ATOM_NAMES[f"c{k}"] = k - 1            # 0..11 caracter t-k
for k in range(1, 9):
    ATOM_NAMES[f"k{k}"] = 11 + k           # 12..19 clase de caracter t-k
ATOM_NAMES.update(dict(wp=20, pw1=21, pw2=22, lpos=23, wlen=24, lfirst=25, pw1f=26, sp=27))
# v0.1 (ver ixchel/core.py): wp hash del prefijo de palabra; pw1/pw2 hash de palabras previas; lpos chars desde
# salto de linea (tope 63); wlen largo del prefijo (tope 15); lfirst primer char de la linea; pw1f primer char de
# la palabra previa; sp = 1 tras linea vacia.
# lexstruct (ronda 2):
ATOM_NAMES.update(dict(
    spk=28,    # hash de la ultima cabecera de hablante (linea en MAYUSCULAS que termina en '.', largo<=40)
    spos=29,   # lineas desde la ultima cabecera de hablante (tope 3)
    plend=30,  # ultimo caracter de la linea anterior (255 si ninguno)
    pl1f=31,   # primer caracter de la linea anterior
    pls=32,    # numero aproximado de silabas (grupos de vocales) de la linea anterior (tope 15)
    lsyl=33,   # silabas aproximadas ya vistas en la linea actual (tope 15)
    wcs=34,    # estado de mayusculas de la palabra en curso: 0 vacia,1 minus,2 Cap/letra sola,3 MAYUSC,4 mixta
    pwcs=35,   # estado de mayusculas de la palabra previa completa
))
NATOM = 36


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


def build_case_flags(itos):
    """bit0 mayuscula, bit1 minuscula, bit2 vocal, bit3 permitido en cabecera de hablante (A-Z, espacio, . ' -), bit4 punto."""
    cf = np.zeros(256, dtype=np.uint8)
    for i, ch in enumerate(itos):
        f = 0
        if ch.isupper():
            f |= 1
        if ch.islower():
            f |= 2
        if ch in "aeiouAEIOU":
            f |= 4
        if ch.isupper() or ch in " .'-":
            f |= 8
        if ch == ".":
            f |= 16
        cf[i] = f
    return cf


@nb.njit(inline="always")
def _wc(wlen, fu, anyl, anyu2):
    # 0 vacia; 1 minuscula inicial; 2 Capitalizada o letra sola; 3 TODO MAYUSCULAS (>=2); 4 mixta
    if wlen == 0:
        return 0
    if fu == 0:
        return 4 if anyu2 == 1 else 1
    if wlen == 1:
        return 2
    if anyl == 0:
        return 3
    return 2 if anyu2 == 0 else 4


@nb.njit(cache=True)
def _atoms(ids, cls, isletter, isnl, cf, cols, out):
    n = ids.shape[0]
    nc = cols.shape[0]
    a = np.zeros(36, dtype=np.int64)
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
    # estado lexstruct
    lh = np.uint64(0)
    lup = 1
    lastch = 255
    lsyl = 0
    invow = 0
    spk = np.uint64(0)
    sl = 1000
    plend = 255
    pl1f = 255
    pls = 0
    fu = 0
    anyl = 0
    anyu2 = 0
    pwcs = 0
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
        a[28] = np.int64(spk >> np.uint64(2))
        a[29] = sl if sl < 3 else 3
        a[30] = plend
        a[31] = pl1f
        a[32] = pls if pls < 15 else 15
        a[33] = lsyl if lsyl < 15 else 15
        a[34] = _wc(wlen, fu, anyl, anyu2)
        a[35] = pwcs
        for j in range(nc):
            out[t, j] = a[cols[j]]
        # consumir ids[t]
        cu = np.int64(ids[t])
        if isletter[cu] == 1:
            if wlen == 0:
                wfirst = cu
                fu = 1 if (cf[cu] & 1) else 0
                anyl = 1 if (cf[cu] & 2) else 0
                anyu2 = 0
            else:
                if (cf[cu] & 1) != 0:
                    anyu2 = 1
                if (cf[cu] & 2) != 0:
                    anyl = 1
            wp = (wp + np.uint64(cu) + np.uint64(1)) * P
            wlen += 1
        else:
            if wlen > 0:
                pw2 = pw1
                pw1 = wp
                pw1f = wfirst
                pwcs = _wc(wlen, fu, anyl, anyu2)
            wp = np.uint64(0)
            wlen = 0
        if isnl[cu] == 1:
            if lpos == 0:
                sp = 1
            else:
                sp = 0
            # cabecera de hablante: linea solo en mayusculas/espacio/. ' -, empieza en mayuscula, termina en '.', largo<=40
            is_h = (lpos > 0) and (lpos <= 40) and (lup == 1) and (lastch < 255) and (lfirst < 255)
            if is_h:
                is_h = ((cf[lastch] & 16) != 0) and ((cf[lfirst] & 1) != 0)
            if is_h:
                spk = lh
                sl = 0
            else:
                if sl < 1000:
                    sl += 1
            plend = lastch
            pl1f = lfirst
            pls = lsyl
            lh = np.uint64(0)
            lup = 1
            lastch = 255
            lsyl = 0
            invow = 0
            lpos = 0
            lfirst = 255
        else:
            if lpos == 0:
                lfirst = cu
            lpos += 1
            lh = (lh + np.uint64(cu) + np.uint64(1)) * P
            if (cf[cu] & 8) == 0:
                lup = 0
            lastch = cu
            v = 1 if (cf[cu] & 4) != 0 else 0
            if v == 1 and invow == 0:
                lsyl += 1
            invow = v


@nb.njit(inline="always")
def _mix(h):
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


@nb.njit(cache=True, parallel=True)
def _level_hashes(A, pos, colidx, lvl_off, out):
    # A [n, ncols] int64 ; pos [m] ; colidx concat of columns-in-A per level ; lvl_off [L1+1]
    L1 = lvl_off.shape[0] - 1
    m = pos.shape[0]
    for i in nb.prange(m):
        t = pos[i]
        for l in range(L1):
            h = np.uint64(0x9E3779B97F4A7C15) * np.uint64(l + 1)
            for q in range(lvl_off[l], lvl_off[l + 1]):
                h = _mix(h ^ (np.uint64(A[t, colidx[q]]) + np.uint64(0x632BE59BD9B4E019)))
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


def build_chain_tables(H, x):
    """H [L1, m] uint64 hashes de contexto por nivel, x [m] uint8 simbolo siguiente.
    Devuelve lista de tablas (una por nivel): nivel < L usa conteos de continuacion, nivel L usa crudos.
    REQUISITO: cada nivel debe ser un CONTEXTO MAS GRUESO (subconjunto de atomos, en el mismo orden) del nivel siguiente."""
    L1 = H.shape[0]
    xs = x.astype(np.uint64)
    tabs = [None] * L1
    for l in range(L1 - 1, -1, -1):
        if l == L1 - 1:
            pk = (H[l] & MASK) | xs
            upk, cnt = np.unique(pk, return_counts=True)
            tabs[l] = _make_table(upk, cnt)
            pk_child = pk
            child_H = H[l]
            upk_c, first = np.unique(pk, return_index=True)
            child_first = first
            child_upk = upk_c
        else:
            ph = H[l][child_first]
            ppk = (ph & MASK) | (child_upk & np.uint64(127))
            uppk, ccnt = np.unique(ppk, return_counts=True)
            tabs[l] = _make_table(uppk, ccnt)
            pk_l = (H[l] & MASK) | xs
            child_upk, child_first = np.unique(pk_l, return_index=True)
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


def lex_chains():
    """Cadenas candidatas de lexstruct. Cada una cumple la relacion de refinamiento (nivel i subconjunto de nivel i+1)."""
    return dict(
        Pw=[[], ["spk"], ["spk", "wp"]],                                   # (hablante, wp)
        Pc=[[], ["spk"], ["spk", "c1"], ["spk", "c1", "c2", "c3"]],        # (hablante, c1..c3)
        Ps=[[], ["spk", "spos"], ["spk", "spos", "c1", "c2"]],             # posicion en el discurso
        Ve=[[], ["plend"], ["plend", "lpos"], ["plend", "lpos", "c1", "c2"]],  # final de linea previa
        Vf=[[], ["pl1f"], ["pl1f", "lpos", "c1"]],                         # primer char de la linea previa
        Vm=[[], ["pls"], ["pls", "lsyl"], ["pls", "lsyl", "c1", "c2"]],    # metrica: silabas
        W2=[[], ["wp"], ["wp", "pw2"]],                                    # (pw2, wp) saltando pw1
        WL=[[], ["wlen"], ["wlen", "c1", "c2"]],                           # largo de palabra + chars
        CS=[[], ["pwcs"], ["pwcs", "wcs"], ["pwcs", "wcs", "wp"]],         # mayusculas palabra previa/actual
        Ka=[[], ["c2"], ["c2", "c3"], ["c2", "c3", "c4"]],                 # contexto saltado (c2..c4)
        Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],           # contexto saltado (c1,c3)
        Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],  # disperso alto orden
    )


class Ixchel:
    def __init__(self, chains=None, gate_chains=("A", "W"), name="Ixchel-v0.1", em_iters=60, prior_m=1.0,
                 dev_frac=0.05, dev_positions=150_000, gated=True, verbose=True, t0=16):
        self.chains = chains if chains is not None else default_chains()
        self.names = list(self.chains.keys())
        self.gate_chains = [self.names.index(c) for c in gate_chains]
        self.name = name
        self.em_iters, self.prior_m, self.dev_frac, self.dev_positions = em_iters, prior_m, dev_frac, dev_positions
        self.gated, self.verbose, self.t0 = gated, verbose, t0
        itos = json.load(open(os.path.join(DATA, "vocab.json")))
        self.cls, self.isletter, self.isnl = build_char_tables(itos)
        self.cf = build_case_flags(itos)
        used = sorted({ATOM_NAMES[a] for ch in self.chains.values() for lvl in ch for a in lvl})
        self.cols = np.array(used, dtype=np.int64)
        col_pos = {c: i for i, c in enumerate(used)}
        self.chain_idx = {}
        for nm, ch in self.chains.items():
            colidx = [col_pos[ATOM_NAMES[a]] for lvl in ch for a in lvl]
            off = [0]
            for lvl in ch:
                off.append(off[-1] + len(lvl))
            self.chain_idx[nm] = (np.array(colidx, np.int64), np.array(off, np.int64))
        self.tabs = {}
        self.w = None

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _atoms_for(self, ids):
        out = np.zeros((len(ids), len(self.cols)), dtype=np.int64)
        _atoms(ids, self.cls, self.isletter, self.isnl, self.cf, self.cols, out)
        return out

    def _H(self, A, pos, nm):
        colidx, off = self.chain_idx[nm]
        H = np.zeros((len(off) - 1, len(pos)), dtype=np.uint64)
        _level_hashes(A, pos, colidx, off, H)
        return H

    def _count(self, A, x, lo, hi):
        pos = np.arange(lo, hi, dtype=np.int64)
        xs = x[lo:hi]
        tabs = {}
        for nm in self.names:
            H = self._H(A, pos, nm)
            tabs[nm] = _concat_tables(build_chain_tables(H, xs))
            del H
        return tabs

    def _chain_dists(self, tabs, A, pos, V, want_p=True):
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

    def _gate(self, LH, NN):
        g = np.zeros(len(LH[0]), dtype=np.int64)
        for ci in self.gate_chains:
            L1 = len(self.chains[self.names[ci]])
            f = (LH[ci] + 1) * 4 + _nbucket(NN[ci])
            g = g * ((L1 + 1) * 4) + f
        return g

    def _G(self):
        G = 1
        for ci in self.gate_chains:
            G *= (len(self.chains[self.names[ci]]) + 1) * 4
        return G

    def fit(self, train, V):
        self.V = V
        t_start = time.time()
        n = len(train)
        A = self._atoms_for(train)
        cut = int(n * (1 - self.dev_frac))
        self.log(f"[{self.name}] contando sobre train[:{cut}] ...")
        tabs = self._count(A, train, self.t0, cut)
        dev_pos = np.linspace(cut, n - 1, min(self.dev_positions, n - cut)).astype(np.int64)
        P, LH, NN = self._chain_dists(tabs, A, dev_pos, V)
        xd = train[dev_pos]
        ptrue = np.stack([p[np.arange(len(dev_pos)), xd] for p in P], axis=1)
        gate = self._gate(LH, NN) if self.gated else np.zeros(len(dev_pos), np.int64)
        G = self._G() if self.gated else 1
        self.dev_bpc_chains = {nm: float(-np.log2(np.maximum(ptrue[:, k], 1e-12)).mean()) for k, nm in enumerate(self.names)}
        self.w = self._em(ptrue, gate, G)
        mix = (self.w[gate] * ptrue).sum(1)
        self.dev_bpc_mix = float(-np.log2(np.maximum(mix, 1e-12)).mean())
        self.log(f"[{self.name}] dev bpc cadenas: {self.dev_bpc_chains}  mezcla: {self.dev_bpc_mix:.4f}  (t={time.time()-t_start:.1f}s)")
        del P
        self.tabs = self._count(A, train, self.t0, n)
        self.fit_time = time.time() - t_start
        self.log(f"[{self.name}] listo en {self.fit_time:.1f}s ; params(pares)={self.n_params():,}")

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
        return int(sum(len(t["psym"]) for tb in self.tabs.values() for t in [tb]))

    def predict(self, ids, pos):
        pos = np.asarray(pos, dtype=np.int64)
        A = self._atoms_for(ids)
        P, LH, NN = self._chain_dists(self.tabs, A, pos, self.V)
        gate = self._gate(LH, NN) if self.gated else np.zeros(len(pos), np.int64)
        W = self.w[gate]
        out = np.zeros((len(pos), self.V))
        for k in range(len(P)):
            out += W[:, k:k + 1] * P[k]
        out = np.maximum(out, 1e-9)
        return (out / out.sum(1, keepdims=True)).astype(np.float32)

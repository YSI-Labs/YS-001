"""IxchelEngine: motor de inferencia INCREMENTAL para Ixchel v0.3 (config 'v03': cadenas A,W,S,L,Kb,Kc,WL,W2 +
cache en contexto con ventanas 256/1024/4096/inf + mezcla geometrica global con pesos poe_w).

Equivalencia: step() reproduce Ixchel.predict(ids, pos) del nucleo (copia en esta carpeta, sin cambios de logica)
para cualquier prefijo. Diferencias: solo redondeo de coma flotante (suma secuencial vs pairwise de numpy,
log de numba vs numpy); se mide en verify.py.

Estado por sesion (incremental, O(1) amortizado por caracter):
  - hist (uint8): historial consumido.
  - hs (uint64[3]): hash de palabra actual wp, hash de las 2 palabras previas pw1, pw2.
  - st (int64[8]): wlen, lpos, lfirst, pw1f, wfirst, sp, n (longitud), used (entradas de la tabla hash).
  - tabla hash abierta (hkeys int64, hvals int32[cap, nW]): conteos por (orden k, contexto, simbolo) para cada
    ventana. Insercion del caracter t: +1 en todas las ventanas (k=1..4). Expiracion: al consumir t, el
    caracter t-W sale de la ventana W (-1). Asi el cache WB de ventana W es exacto sin recalcular indices.
Modelo (solo lectura, mmap, compartible entre procesos): tablas de cadenas KN empaquetadas + unigrama + pesos.
Kernel por token en numba (1 hilo). El estado de cadenas no depende del historial: solo atomos (O(1)).
"""
import os, json, time
import numpy as np
import numba as nb

ATOM = {}
for _k in range(1, 13):
    ATOM[f"c{_k}"] = _k - 1
for _k in range(1, 9):
    ATOM[f"k{_k}"] = 11 + _k
ATOM.update(dict(wp=20, pw1=21, pw2=22, lpos=23, wlen=24, lfirst=25, pw1f=26, sp=27))

SEED = np.uint64(0x9E3779B97F4A7C15)
ADDC = np.uint64(0x632BE59BD9B4E019)
MC1 = np.uint64(0xBF58476D1CE4E5B9)
MC2 = np.uint64(0x94D049BB133111EB)
PMUL = np.uint64(1099511628211)
U1 = np.uint64(1)
U2 = np.uint64(2)
U7 = np.uint64(7)
U27 = np.uint64(27)
U30 = np.uint64(30)
U31 = np.uint64(31)


# ============================================================ empaquetado (solo al guardar)
def pack_model(outdir, tabs, base, meta, cls, isl, isn):
    """tabs: dict nm -> dict(ckey,cstart,psym,pcnt,c_off,cs_off,D) como en core_v02.Ixchel.tabs.
    Concatena todas las cadenas en arrays globales con offsets absolutos."""
    os.makedirs(outdir, exist_ok=True)
    names = list(meta["chains"].keys())
    ck, cs, ps, pc, coff, csoff, D = [], [], [], [], [], [], []
    ckb = psb = csb = lv = 0
    CK_BASE, PS_BASE, CS_BASE, LVL0, NLVL, COLS, COLOFF = [], [], [], [], [], [], []
    colpos = 0
    for nm in names:
        T = tabs[nm]
        L1 = len(T["D"])
        CK_BASE.append(ckb); PS_BASE.append(psb); CS_BASE.append(csb); LVL0.append(lv); NLVL.append(L1)
        ck.append(T["ckey"]); ps.append(T["psym"]); pc.append(T["pcnt"])
        cs.append(np.asarray(T["cstart"], np.int64) + psb)
        coff.append(np.asarray(T["c_off"][:-1], np.int64) + ckb)
        csoff.append(np.asarray(T["cs_off"][:-1], np.int64) + csb)
        D.append(np.asarray(T["D"], np.float64))
        for lvl in meta["chains"][nm]:
            COLOFF.append(colpos)
            for a in lvl:
                COLS.append(ATOM[a]); colpos += 1
        ckb += len(T["ckey"]); psb += len(T["psym"]); csb += len(T["cstart"]); lv += L1
    COLOFF.append(colpos)
    coff.append(np.array([ckb], np.int64)); csoff.append(np.array([csb], np.int64))
    arrs = dict(GCK=np.concatenate(ck).astype(np.uint64), GCS=np.concatenate(cs).astype(np.int64),
                GPS=np.concatenate(ps).astype(np.uint8), GPC=np.concatenate(pc).astype(np.int32),
                GCOFF=np.concatenate(coff).astype(np.int64), GCSOFF=np.concatenate(csoff).astype(np.int64),
                GD=np.concatenate(D).astype(np.float64), LVL0=np.array(LVL0, np.int64),
                NLVL=np.array(NLVL, np.int64), COLS=np.array(COLS, np.int64),
                COLOFF=np.array(COLOFF, np.int64), base=np.asarray(base, np.float64),
                poe_w=np.asarray(meta["poe_w"], np.float64),
                windows=np.asarray(meta["cache_windows"], np.int64),
                cls=np.asarray(cls, np.uint8), isletter=np.asarray(isl, np.uint8), isnl=np.asarray(isn, np.uint8),
                CK_BASE=np.array(CK_BASE, np.int64), PS_BASE=np.array(PS_BASE, np.int64),
                CS_BASE=np.array(CS_BASE, np.int64))
    for k, v in arrs.items():
        np.save(os.path.join(outdir, k + ".npy"), v)
    meta2 = {k: v for k, v in meta.items()}
    meta2["names"] = names
    meta2["V"] = int(meta["V"])
    json.dump(meta2, open(os.path.join(outdir, "meta.json"), "w"), indent=1)


# ============================================================ kernels numba
@nb.njit(cache=True, inline="always")
def _mix(h):
    h ^= h >> U30
    h *= MC1
    h ^= h >> U27
    h *= MC2
    h ^= h >> U31
    return h


@nb.njit(cache=True)
def build_level_hash(GCK, GCOFF):
    """Indice hash abierto sobre (nivel gl, clave): HK[slot] = indice global en GCK (int32), -1 vacio.
    Sustituye la busqueda binaria de core_v02._chain_predict por una busqueda O(1) esperada; mismo resultado
    porque las claves son unicas dentro de cada nivel."""
    N = GCK.shape[0]
    cap = 1
    while cap < 2 * N:
        cap *= 2
    HK = np.full(cap, -1, np.int32)
    mask = cap - 1
    for gl in range(GCOFF.shape[0] - 1):
        for idx in range(GCOFF[gl], GCOFF[gl + 1]):
            i = np.int64(_mix(GCK[idx] ^ (np.uint64(gl + 1) * SEED)) & np.uint64(mask))
            while HK[i] != -1:
                i = (i + 1) & mask
            HK[i] = np.int32(idx)
    return HK


@nb.njit(cache=True)
def _atoms_at(hist, t, hs, st, cls, a):
    """Vector de 28 atomos en la posicion t (solo usa hist[:t]); identico a core_v02._atoms."""
    for k in range(12):
        j = t - k - 1
        if j >= 0:
            c = np.int64(hist[j])
        else:
            c = np.int64(255)
        a[k] = c
        if k < 8:
            if c < 255:
                a[12 + k] = np.int64(cls[c])
            else:
                a[12 + k] = np.int64(0)
    a[20] = np.int64(hs[0] >> U2)
    a[21] = np.int64(hs[1] >> U2)
    a[22] = np.int64(hs[2] >> U2)
    lp = st[1]
    a[23] = lp if lp < 63 else 63
    wl = st[0]
    a[24] = wl if wl < 15 else 15
    a[25] = st[2]
    a[26] = st[3]
    a[27] = st[5]


@nb.njit(cache=True)
def _ht_find(hkeys, key):
    mask = hkeys.shape[0] - 1
    i = np.int64(_mix(np.uint64(key)) & np.uint64(mask))
    while True:
        kk = hkeys[i]
        if kk == key:
            return i
        if kk == -1:
            return -1
        i = (i + 1) & mask


@nb.njit(cache=True)
def _ht_slot(hkeys, key, st):
    mask = hkeys.shape[0] - 1
    i = np.int64(_mix(np.uint64(key)) & np.uint64(mask))
    while True:
        kk = hkeys[i]
        if kk == key:
            return i
        if kk == -1:
            hkeys[i] = key
            st[7] += 1
            return i
        i = (i + 1) & mask


@nb.njit(cache=True)
def _ht_rebuild(hkeys, hvals, newcap):
    nW = hvals.shape[1]
    nk = np.full(newcap, -1, np.int64)
    nv = np.zeros((newcap, nW), np.int32)
    mask = newcap - 1
    for i in range(hkeys.shape[0]):
        key = hkeys[i]
        if key != -1:
            j = np.int64(_mix(np.uint64(key)) & np.uint64(mask))
            while nk[j] != -1:
                j = (j + 1) & mask
            nk[j] = key
            for w in range(nW):
                nv[j, w] = hvals[i, w]
    return nk, nv


@nb.njit(cache=True)
def _consume(hist, c, hs, st, isletter, isnl, windows, hkeys, hvals):
    """Consume el caracter c en la posicion t = st[6]: actualiza cache (insercion + expiracion) y atomos."""
    t = st[6]
    hist[t] = c
    nW = windows.shape[0]
    cc = np.int64(c)
    # (1) insertar entradas (orden k, contexto x[t-k:t], simbolo x[t]) en todas las ventanas
    for k in range(1, 5):
        if t < k:
            break
        code = np.int64(0)
        for j in range(k):
            code = code * 128 + np.int64(hist[t - k + j])
        key = (np.int64(k) << 36) | (code << 7) | cc
        slot = _ht_slot(hkeys, key, st)
        for w in range(nW):
            hvals[slot, w] += 1
    # (2) expirar la entrada t-W de cada ventana finita W (sale de la ventana al avanzar a t+1)
    for w in range(nW):
        tp = t - windows[w]
        for k in range(1, 5):
            if tp < k:
                break
            code = np.int64(0)
            for j in range(k):
                code = code * 128 + np.int64(hist[tp - k + j])
            key = (np.int64(k) << 36) | (code << 7) | np.int64(hist[tp])
            slot = _ht_find(hkeys, key)
            hvals[slot, w] -= 1
    # (3) estado de atomos (mismo orden de actualizacion que core_v02._atoms)
    if isletter[c] == 1:
        if st[0] == 0:
            st[4] = cc
        hs[0] = (hs[0] + np.uint64(c) + U1) * PMUL
        st[0] += 1
    else:
        if st[0] > 0:
            hs[2] = hs[1]
            hs[1] = hs[0]
            st[3] = st[4]
        hs[0] = np.uint64(0)
        st[0] = 0
    if isnl[c] == 1:
        st[5] = 1 if st[1] == 0 else 0
        st[1] = 0
        st[2] = 255
    else:
        if st[1] == 0:
            st[2] = cc
        st[1] += 1
    st[6] = t + 1


@nb.njit(cache=True)
def _feed(hist, ids, hs, st, isletter, isnl, windows, hkeys, hvals):
    for i in range(ids.shape[0]):
        _consume(hist, ids[i], hs, st, isletter, isnl, windows, hkeys, hvals)


@nb.njit(cache=True)
def _dist(hist, st, hs, cls, GCK, GCS, GPS, GPC, GCOFF, GCSOFF, GD, LVL0, NLVL, COLS, COLOFF, HK,
          base, poe_w, windows, hkeys, hvals, out32):
    """Distribucion p(. | hist[:n]) como float32 [V]; replica core_v02.Ixchel.predict en modo poe."""
    V = base.shape[0]
    NCH = LVL0.shape[0]
    nW = windows.shape[0]
    NE = NCH + nW
    n = st[6]
    a = np.zeros(28, np.int64)
    _atoms_at(hist, n, hs, st, cls, a)
    LP = np.empty((NE, V))
    # ---- cadenas KN (tablas fijas): nivel a nivel, el mas fino encontrado manda la ultima multiplicacion
    for ci in range(NCH):
        p = np.full(V, 1.0 / V)
        for l in range(NLVL[ci]):
            gl = LVL0[ci] + l
            h = SEED * np.uint64(l + 1)
            for q in range(COLOFF[gl], COLOFF[gl + 1]):
                h = _mix(h ^ (np.uint64(a[COLS[q]]) + ADDC))
            key = _mix(h) >> U7
            lo = GCOFF[gl]
            hi = GCOFF[gl + 1]
            mask = HK.shape[0] - 1
            i = np.int64(_mix(key ^ (np.uint64(gl + 1) * SEED)) & np.uint64(mask))
            aa = -1
            while True:
                idx = np.int64(HK[i])
                if idx < 0:
                    break
                if idx >= lo and idx < hi and GCK[idx] == key:
                    aa = idx
                    break
                i = (i + 1) & mask
            if aa < 0:
                break
            j = aa - lo
            s = GCS[GCSOFF[gl] + j]
            e = GCS[GCSOFF[gl] + j + 1]
            nn = 0.0
            N1 = 0.0
            N2 = 0.0
            N3 = 0.0
            for q in range(s, e):
                c = GPC[q]
                nn += c
                if c == 1:
                    N1 += 1.0
                elif c == 2:
                    N2 += 1.0
                else:
                    N3 += 1.0
            gamma = (GD[gl, 0] * N1 + GD[gl, 1] * N2 + GD[gl, 2] * N3) / nn
            for x in range(V):
                p[x] *= gamma
            for q in range(s, e):
                c = GPC[q]
                if c == 1:
                    d = GD[gl, 0]
                elif c == 2:
                    d = GD[gl, 1]
                else:
                    d = GD[gl, 2]
                p[GPS[q]] += (c - d) / nn
        for x in range(V):
            LP[ci, x] = np.log(max(p[x], 1e-12))
    # ---- cache WB por ventana desde la tabla incremental
    P = np.empty((nW, V))
    cnt = np.zeros((nW, V))
    tot = np.zeros(nW)
    nd = np.zeros(nW)
    for w in range(nW):
        for v in range(V):
            P[w, v] = base[v]
    for k in range(1, 5):
        if n < k:
            break
        code = np.int64(0)
        for j in range(k):
            code = code * 128 + np.int64(hist[n - k + j])
        for w in range(nW):
            tot[w] = 0.0
            nd[w] = 0.0
        for v in range(V):
            key = (np.int64(k) << 36) | (code << 7) | np.int64(v)
            slot = _ht_find(hkeys, key)
            for w in range(nW):
                cw = np.int64(0)
                if slot >= 0:
                    cw = np.int64(hvals[slot, w])
                cnt[w, v] = cw
                if cw > 0:
                    tot[w] += cw
                    nd[w] += 1.0
        for w in range(nW):
            if tot[w] > 0.0:
                td = nd[w]
                den = tot[w] + td
                for v in range(V):
                    P[w, v] = (cnt[w, v] + td * P[w, v]) / den
    for w in range(nW):
        for v in range(V):
            LP[NCH + w, v] = np.log(max(P[w, v], 1e-12))
    # ---- mezcla geometrica global (poe_combine)
    L = np.zeros(V)
    for e in range(NE):
        we = poe_w[e]
        for v in range(V):
            L[v] += we * LP[e, v]
    mx = L.max()
    s = 0.0
    for v in range(V):
        L[v] = np.exp(L[v] - mx)
        s += L[v]
    for v in range(V):
        out32[v] = np.float32(L[v] / s)


@nb.njit(cache=True)
def _sample(dist, u):
    """Muestreo como numpy Generator.choice(V, p): cdf = cumsum(p/sum); cdf/=cdf[-1]; searchsorted(u, 'right')."""
    V = dist.shape[0]
    tot = 0.0
    for v in range(V):
        tot += np.float64(dist[v])
    cdf = np.empty(V)
    acc = 0.0
    for v in range(V):
        acc += np.float64(dist[v]) / tot
        cdf[v] = acc
    for v in range(V):
        cdf[v] /= cdf[V - 1]
    idx = 0
    while idx < V - 1 and cdf[idx] <= u:
        idx += 1
    return idx


@nb.njit(cache=True)
def _score_stream(hist, st, hs, cls, GCK, GCS, GPS, GPC, GCOFF, GCSOFF, GD, LVL0, NLVL, COLS, COLOFF, HK,
                  base, poe_w, windows, hkeys, hvals, ids, pos, isletter, isnl, out32_all):
    """Puntuacion por lotes: consume ids en orden y, en cada pos[i] (ascendente), guarda la distribucion."""
    j = 0
    for t in range(ids.shape[0]):
        while j < pos.shape[0] and pos[j] == t:
            _dist(hist, st, hs, cls, GCK, GCS, GPS, GPC, GCOFF, GCSOFF, GD, LVL0, NLVL, COLS, COLOFF, HK,
                  base, poe_w, windows, hkeys, hvals, out32_all[j])
            j += 1
        _consume(hist, ids[t], hs, st, isletter, isnl, windows, hkeys, hvals)


@nb.njit(cache=True)
def _gen_loop(hist, st, hs, cls, GCK, GCS, GPS, GPC, GCOFF, GCSOFF, GD, LVL0, NLVL, COLS, COLOFF, HK,
              base, poe_w, windows, hkeys, hvals, isletter, isnl, us, out_ids, out32):
    """Generacion completa en numba: dist + muestreo (con uniformes precalculados) + consumo, por token."""
    for i in range(us.shape[0]):
        _dist(hist, st, hs, cls, GCK, GCS, GPS, GPC, GCOFF, GCSOFF, GD, LVL0, NLVL, COLS, COLOFF, HK,
              base, poe_w, windows, hkeys, hvals, out32)
        c = _sample(out32, us[i])
        out_ids[i] = c
        _consume(hist, np.uint8(c), hs, st, isletter, isnl, windows, hkeys, hvals)


# ============================================================ API
class Session:
    """Estado de una conversacion/prefijo. Todo lo necesario para continuar la generacion en O(1)/caracter."""

    def __init__(self, nW, cap=1024, hcap=1 << 12):
        self.hist = np.zeros(cap, np.uint8)
        self.hs = np.zeros(3, np.uint64)
        self.st = np.zeros(8, np.int64)
        self.st[2] = 255; self.st[3] = 255; self.st[4] = 255
        self.hkeys = np.full(hcap, -1, np.int64)
        self.hvals = np.zeros((hcap, nW), np.int32)

    @property
    def n(self):
        return int(self.st[6])

    def nbytes(self):
        return int(self.hist.nbytes + self.hs.nbytes + self.st.nbytes + self.hkeys.nbytes + self.hvals.nbytes)


class IxchelEngine:
    def __init__(self, path):
        meta = json.load(open(os.path.join(path, "meta.json")))
        self.meta = meta
        self.V = int(meta["V"])
        self.itos = meta["itos"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.names = meta["names"]
        self.path = path
        ld = lambda k: np.load(os.path.join(path, k + ".npy"), mmap_mode="r")
        self.A = {k: ld(k) for k in ("GCK", "GCS", "GPS", "GPC", "GCOFF", "GCSOFF", "GD", "LVL0", "NLVL",
                                     "COLS", "COLOFF", "HK", "base", "poe_w", "windows", "cls", "isletter", "isnl")}
        self.nW = len(self.A["windows"])
        self.kstate = (self.A["cls"], self.A["GCK"], self.A["GCS"], self.A["GPS"], self.A["GPC"],
                       self.A["GCOFF"], self.A["GCSOFF"], self.A["GD"], self.A["LVL0"], self.A["NLVL"],
                       self.A["COLS"], self.A["COLOFF"], self.A["HK"], self.A["base"], self.A["poe_w"],
                       self.A["windows"])
        self._out = np.empty(self.V, np.float32)

    @classmethod
    def load(cls, path):
        return cls(path)

    def n_params(self):
        return int(len(self.A["GPS"]))

    def model_nbytes(self):
        return int(sum(v.nbytes for v in self.A.values()))

    def _enc(self, prompt):
        if isinstance(prompt, str):
            return np.array([self.stoi[c] for c in prompt], np.uint8)
        return np.asarray(prompt, np.uint8)

    def _grow(self, sess, extra_chars):
        need = sess.n + extra_chars
        if need > len(sess.hist):
            h = np.zeros(max(need, 2 * len(sess.hist)), np.uint8)
            h[:sess.n] = sess.hist[:sess.n]
            sess.hist = h
        need_h = int(sess.st[7]) + 4 * extra_chars
        if need_h * 2 > len(sess.hkeys):
            newcap = len(sess.hkeys)
            while need_h * 2 > newcap:
                newcap *= 2
            sess.hkeys, sess.hvals = _ht_rebuild(sess.hkeys, sess.hvals, newcap)

    def feed(self, sess, ids):
        ids = self._enc(ids)
        self._grow(sess, len(ids))
        A = self.A
        _feed(sess.hist, ids, sess.hs, sess.st, A["isletter"], A["isnl"], A["windows"], sess.hkeys, sess.hvals)
        return sess

    def new_session(self, prompt=""):
        sess = Session(self.nW)
        return self.feed(sess, prompt) if len(prompt) else sess

    def step(self, sess, out=None):
        """Distribucion (float32 [V]) del siguiente caracter dado el historial consumido."""
        out = self._out if out is None else out
        A = self.A
        _dist(sess.hist, sess.st, sess.hs, *self.kstate, sess.hkeys, sess.hvals, out)
        return out

    def generate_fast(self, sess, n, seed=0):
        """Generacion con el bucle completo en numba (un solo despacho Python). Mismo muestreo que generate()."""
        rng = np.random.default_rng(seed)
        us = rng.random(n)
        ids = np.zeros(n, np.uint8)
        self._grow(sess, n)
        A = self.A
        _gen_loop(sess.hist, sess.st, sess.hs, *self.kstate, sess.hkeys, sess.hvals, A["isletter"], A["isnl"],
                  us, ids, self._out)
        return "".join(self.itos[i] for i in ids), ids

    def generate(self, sess, n, seed=0, return_dists=False, timings=None):
        """Muestreo ancestral (temperatura 1, mismo generador que harness.generate). timings: array int64[n] opcional
        donde se guarda la duracion de cada token en ns (dist + muestreo + consumo)."""
        rng = np.random.default_rng(seed)
        ids = np.zeros(n, np.uint8)
        dists = np.zeros((n, self.V), np.float32) if return_dists else None
        self._grow(sess, n)
        A = self.A
        clock = time.perf_counter_ns
        for i in range(n):
            t0 = clock() if timings is not None else 0
            d = self.step(sess, out=self._out)
            if return_dists:
                dists[i] = d
            c = _sample(d, rng.random())
            ids[i] = c
            _consume(sess.hist, np.uint8(c), sess.hs, sess.st, A["isletter"], A["isnl"], A["windows"],
                     sess.hkeys, sess.hvals)
            if timings is not None:
                timings[i] = clock() - t0
        text = "".join(self.itos[i] for i in ids)
        return (text, ids, dists) if return_dists else text

    def score_positions(self, ids, pos):
        """Lote: distribuciones en pos (ascendente) para el texto ids; mismo resultado que step en cada prefijo."""
        ids = self._enc(ids)
        pos = np.asarray(pos, np.int64)
        sess = Session(self.nW, cap=len(ids) + 8, hcap=1 << 12)
        self._grow(sess, len(ids))
        out = np.zeros((len(pos), self.V), np.float32)
        A = self.A
        _score_stream(sess.hist, sess.st, sess.hs, *self.kstate, sess.hkeys, sess.hvals, ids, pos,
                      A["isletter"], A["isnl"], out)
        return out

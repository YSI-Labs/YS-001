"""Tz'ib: modelo de caracteres de orden no acotado sobre un arreglo de sufijos.

Idea: el contexto ids[:pos] se busca en el train como sufijo. Para no recalcular la busqueda
en cada orden, el arreglo de sufijos se construye sobre el train INVERTIDO: extender el
contexto hacia la izquierda = anadir un caracter al patron = refinar un intervalo del SA.
Para una coincidencia de longitud L, las ocurrencias del contexto dan (via SA) el caracter
siguiente y el caracter anterior (para la continuacion de Kneser-Ney).

Distribuciones que se mezclan (todas sin learning rate):
  WB-k   : Witten-Bell interpolado hasta orden k (k=5,12,20,inf) -- mismo esquema que common/baseline_wb.py
  KN-k   : Kneser-Ney interpolado con descuentos D_L = n1/(n1+2 n2) (cerrados, de los datos del train)
  EXCL-k : WB con escape y exclusion de simbolos ya vistos en ordenes superiores (top-down)
Orden 0..2 usan tablas densas; orden >=3 usan escaneo del intervalo del SA.
"""
import os
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import sys, time, zlib
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
DMAX = 24          # descuentos y conteo de k-gramas hasta k=DMAX (orden > DMAX reutiliza D[DMAX-1])
LBUF = 4096        # tope de longitud de coincidencia por posicion (buffer); se registra si se alcanza
INF = 10 ** 9
CAPS = np.array([5, 12, 20, INF], dtype=np.int64)
METHODS = ["WB-5", "WB-12", "WB-20", "WB-inf", "KN-5", "KN-12", "KN-20", "KN-inf", "EXCL-5", "EXCL-inf"]


# ----------------------------------------------------------------- indice (solo con datos de train)
def build_sa(x):
    """Arreglo de sufijos de R = reverse(x) por doubling con argsort estable (numpy)."""
    R = x[::-1].astype(np.int64)
    n = len(R)
    rank = R.copy()
    M = n + 2
    k = 1
    while True:
        r2 = np.zeros(n, np.int64)
        if k < n:
            r2[:n - k] = rank[k:] + 1
        key = rank * M + r2
        sa = np.argsort(key, kind="stable")
        ks = key[sa]
        d = np.zeros(n, np.int64)
        d[1:] = ks[1:] != ks[:-1]
        nr = np.cumsum(d)
        rank = np.empty(n, np.int64)
        rank[sa] = nr
        if nr[-1] == n - 1:
            break
        k *= 2
    return sa.astype(np.int64)


def build_meta(x, V):
    xi = x.astype(np.int64)
    n = len(xi)
    uni = np.bincount(xi, minlength=V).astype(np.float64)
    b = xi[:-1] * V + xi[1:]
    raw1 = np.bincount(b, minlength=V * V).astype(np.float64)
    t3 = xi[:-2] * V * V + xi[1:-1] * V + xi[2:]
    raw2 = np.bincount(t3, minlength=V ** 3).astype(np.float64)
    u3 = np.unique(t3)                       # (w,h,s) distinctos
    cont1 = np.bincount(u3 % (V * V), minlength=V * V).astype(np.float64)
    t4 = xi[:-3] * V ** 3 + xi[1:-2] * V * V + xi[2:-1] * V + xi[3:]
    u4 = np.unique(t4)                       # (w,h1,h2,s) distinctos
    cont2 = np.bincount(u4 % (V ** 3), minlength=V ** 3).astype(np.float64)
    u2 = np.unique(b)                        # (w,s) distinctos
    cont0 = np.bincount(u2 % V, minlength=V).astype(np.float64)
    del u3, u4, t4, t3
    # descuentos cerrados y numero de k-gramas distintos, orden L <-> (L+1)-gramas
    Dk = np.zeros(DMAX + 2)          # Dk[k] = n1/(n1+2 n2) sobre k-gramas crudos
    entries = np.zeros(DMAX + 2, np.int64)
    h = xi.astype(np.uint64)
    for k in range(1, DMAX + 2):
        if k > 1:
            h = h[:-1] * np.uint64(1000003) + xi[k - 1:].astype(np.uint64)
        _, c = np.unique(h, return_counts=True)
        n1 = float((c == 1).sum())
        n2 = float((c == 2).sum())
        Dk[k] = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.0
        entries[k] = len(c)
    # nivel crudo de orden L (contexto de L chars, siguiente char): usa k = L+1 (k-gramas)
    # nivel de continuacion de orden L (cuenta distintos w antes de h s = (L+2)-gramas): usa k = L+2
    Dr = Dk[1:DMAX + 1].copy()
    Dc = Dk[2:DMAX + 2].copy()
    return dict(uni=uni, raw0=uni.copy(), raw1=raw1, raw2=raw2, cont0=cont0, cont1=cont1,
                cont2=cont2, Dr=Dr, Dc=Dc, entries=entries[:DMAX + 1])


def build_index(x, V, cache=True):
    """Devuelve (sa, meta, tiempos). Cachea en experiments/r1_tzib/cache (solo depende del train)."""
    t0 = time.time()
    key = f"{len(x)}_{zlib.crc32(x.tobytes()):08x}_{V}"
    os.makedirs(CACHE, exist_ok=True)
    sa_p = os.path.join(CACHE, f"sa_{key}.npy")
    meta_p = os.path.join(CACHE, f"meta2_{key}.npz")
    if cache and os.path.exists(sa_p) and os.path.exists(meta_p):
        sa = np.load(sa_p)
        meta = dict(np.load(meta_p))
        return sa, meta, dict(hit=True, build_s=time.time() - t0)
    sa = build_sa(x)
    t1 = time.time()
    meta = build_meta(x, V)
    if cache:
        np.save(sa_p, sa)
        np.savez(meta_p, **meta)
    return sa, meta, dict(hit=False, build_s=time.time() - t0, sa_s=t1 - t0)


# ----------------------------------------------------------------- consulta
@njit(cache=True)
def _refine(R, SA, n, lo, hi, off, c):
    """Sub-intervalo de [lo,hi) cuyo caracter en R[SA[i]+off] es c (-1 si el sufijo termina antes)."""
    a = lo
    b = hi
    while a < b:
        m = (a + b) >> 1
        x = SA[m] + off
        kv = -1
        if x < n:
            kv = np.int64(R[x])
        if kv < c:
            a = m + 1
        else:
            b = m
    start = a
    b = hi
    while a < b:
        m = (a + b) >> 1
        x = SA[m] + off
        kv = -1
        if x < n:
            kv = np.int64(R[x])
        if kv <= c:
            a = m + 1
        else:
            b = m
    return start, a


@njit(cache=True)
def _excl(Craw, raw0, Ntot, Ucnt, V, t, p, E):
    """WB con exclusion, top-down desde el orden t hasta 0 (escape WB, simbolos vistos excluidos)."""
    for s in range(V):
        p[s] = 0.0
        E[s] = False
    Rm = 1.0
    for L in range(t, -1, -1):
        if L == 0:
            nonex = 0
            N = 0.0
            U = 0.0
            for s in range(V):
                if not E[s]:
                    nonex += 1
                    N += raw0[s]
                    if raw0[s] > 0:
                        U += 1.0
            if nonex == 0:
                break
            u = 1.0 / nonex
            den = N + U
            for s in range(V):
                if not E[s]:
                    if den > 0:
                        p[s] += Rm * (raw0[s] + U * u) / den
                    else:
                        p[s] += Rm * u
        else:
            N = 0.0
            U = 0.0
            for s in range(V):
                if not E[s]:
                    N += Craw[L, s]
                    if Craw[L, s] > 0:
                        U += 1.0
            den = N + U
            if den > 0:
                for s in range(V):
                    if (not E[s]) and Craw[L, s] > 0:
                        p[s] += Rm * Craw[L, s] / den
                Rm *= U / den
            for s in range(V):
                if Craw[L, s] > 0:
                    E[s] = True


@njit(cache=True)
def _core(R, SA, n, V, ids, pos, uni, U0, raw0, raw1, cont1, raw2, cont2, cont0, Dr, Dc, caps):
    m = pos.shape[0]
    out = np.zeros((m, 10, V))
    lstar = np.zeros(m, np.int64)
    trunc = np.zeros(m, np.bool_)
    Craw = np.zeros((LBUF, V))
    Ccont = np.zeros((LBUF, V))
    P = np.zeros((LBUF, V))
    Q = np.zeros((LBUF, V))
    ilo = np.zeros(LBUF, np.int64)
    ihi = np.zeros(LBUF, np.int64)
    Ntot = np.zeros(LBUF)
    Ucnt = np.zeros(LBUF)
    Acont = np.zeros(LBUF)
    Ucont = np.zeros(LBUF)
    stamp = np.zeros(V * V, np.int64)
    cur = 0
    E = np.zeros(V, np.bool_)
    pex = np.zeros(V)
    nn = float(n)
    for k in range(m):
        p = pos[k]
        # 1) intervalos del SA para ctx[-L:], L = 1..Lint (existencia de la coincidencia)
        ilo[0] = 0
        ihi[0] = n
        Lint = 0
        lim = min(p, LBUF - 1)
        for L in range(1, lim + 1):
            c = np.int64(ids[p - L])
            a, b = _refine(R, SA, n, ilo[L - 1], ihi[L - 1], L - 1, c)
            if a >= b:
                break
            ilo[L] = a
            ihi[L] = b
            Lint = L
        if Lint == LBUF - 1:
            trunc[k] = True
        # 2) conteos: siguiente caracter (crudo) y continuacion (distintos w antes de h s)
        Ls = 0
        for s in range(V):
            Craw[0, s] = raw0[s]
            Ccont[0, s] = cont0[s]
        Ntot[0] = nn
        Ucnt[0] = U0
        ac = 0.0
        uc = 0.0
        for s in range(V):
            ac += cont0[s]
            if cont0[s] > 0:
                uc += 1.0
        Acont[0] = ac
        Ucont[0] = uc
        for L in range(1, Lint + 1):
            for s in range(V):
                Craw[L, s] = 0.0
                Ccont[L, s] = 0.0
            if L == 1:
                key = np.int64(ids[p - 1]) * V
                for s in range(V):
                    Craw[1, s] = raw1[key + s]
                    Ccont[1, s] = cont1[key + s]
            elif L == 2:
                key = (np.int64(ids[p - 2]) * V + np.int64(ids[p - 1])) * V
                for s in range(V):
                    Craw[2, s] = raw2[key + s]
                    Ccont[2, s] = cont2[key + s]
            else:
                cur += 1
                for i in range(ilo[L], ihi[L]):
                    x = SA[i]
                    if x >= 1:
                        s = np.int64(R[x - 1])
                        Craw[L, s] += 1.0
                        if x + L < n:
                            w = np.int64(R[x + L])
                            idx = s * V + w
                            if stamp[idx] != cur:
                                stamp[idx] = cur
                                Ccont[L, s] += 1.0
            tot = 0.0
            uu = 0.0
            ac = 0.0
            uc = 0.0
            for s in range(V):
                tot += Craw[L, s]
                if Craw[L, s] > 0:
                    uu += 1.0
                ac += Ccont[L, s]
                if Ccont[L, s] > 0:
                    uc += 1.0
            if tot <= 0:
                break
            Ls = L
            Ntot[L] = tot
            Ucnt[L] = uu
            Acont[L] = ac
            Ucont[L] = uc
        lstar[k] = Ls
        if Ls == 0:
            for mth in range(10):
                for s in range(V):
                    out[k, mth, s] = 1.0 / V
            continue
        # 3) Witten-Bell interpolado (mismo esquema que baseline_wb)
        for s in range(V):
            P[0, s] = (uni[s] + U0 / V) / (nn + U0)
        for L in range(1, Ls + 1):
            N = Ntot[L]
            U = Ucnt[L]
            for s in range(V):
                P[L, s] = (Craw[L, s] + U * P[L - 1, s]) / (N + U)
        # 4) Kneser-Ney: cadena de continuacion Q[L] (L < orden superior), orden superior con crudos
        D0 = Dc[0]
        for s in range(V):
            Q[0, s] = max(Ccont[0, s] - D0, 0.0) / Acont[0] + D0 * Ucont[0] / Acont[0] / V
        for L in range(1, Ls):
            A = Acont[L]
            dL = Dc[min(L, DMAX - 1)]
            if A > 0:
                for s in range(V):
                    Q[L, s] = max(Ccont[L, s] - dL, 0.0) / A + dL * Ucont[L] / A * Q[L - 1, s]
            else:
                for s in range(V):
                    Q[L, s] = Q[L - 1, s]
        for ci in range(4):
            t = min(Ls, caps[ci])
            for s in range(V):
                out[k, ci, s] = P[t, s]
            N = Ntot[t]
            U = Ucnt[t]
            dt = Dr[min(t, DMAX - 1)]
            for s in range(V):
                out[k, 4 + ci, s] = max(Craw[t, s] - dt, 0.0) / N + dt * U / N * Q[t - 1, s]
        # 5) WB con exclusion
        _excl(Craw, raw0, Ntot, Ucnt, V, min(Ls, 5), pex, E)
        for s in range(V):
            out[k, 8, s] = pex[s]
        _excl(Craw, raw0, Ntot, Ucnt, V, Ls, pex, E)
        for s in range(V):
            out[k, 9, s] = pex[s]
    return out, lstar, trunc


class Tzib:
    """Contrato harness: fit(train,V), predict(ids,pos), n_params(), name.
    variant in METHODS. train_frac<1 entrena con un prefijo del train (curva de escalado de datos)."""

    def __init__(self, variant="WB-inf", train_frac=1.0, cache=True):
        assert variant in METHODS, variant
        self.variant = variant
        self.mi = METHODS.index(variant)
        self.train_frac = train_frac
        self.cache = cache
        self.name = f"tzib-{variant}" + ("" if train_frac == 1.0 else f"-f{train_frac:g}")
        self.last_lstar = None
        self.last_trunc = 0
        self.fit_info = None

    def fit(self, train, V):
        x = np.ascontiguousarray(train, dtype=np.uint8)
        if self.train_frac < 1.0:
            x = np.ascontiguousarray(x[: int(len(x) * self.train_frac)])
        self.V = V
        self.n = len(x)
        self.R = np.ascontiguousarray(x[::-1])
        self.sa, self.meta, self.fit_info = build_index(x, V, cache=self.cache)
        m = self.meta
        self.U0 = float((m["uni"] > 0).sum())
        self.caps = CAPS

    def _args(self):
        m = self.meta
        return (self.R, self.sa, self.n, self.V)

    def predict(self, ids, pos):
        pos = np.ascontiguousarray(pos, dtype=np.int64)
        m = self.meta
        ids = np.ascontiguousarray(ids, dtype=np.uint8)
        out, ls, tr = _core(self.R, self.sa, self.n, self.V, ids, pos, m["uni"], self.U0, m["raw0"],
                            m["raw1"], m["cont1"], m["raw2"], m["cont2"], m["cont0"], m["Dr"], m["Dc"], self.caps)
        self.last_lstar = ls
        self.last_trunc = int(tr.sum())
        return out[:, self.mi, :].astype(np.float32)

    def n_params(self):
        """Celdas de conteo comparables con baseline_wb: k-gramas distintos hasta el orden del cap + V.
        (Para orden ilimitado se cuentan hasta DMAX; el arreglo de sufijos, n enteros, se reporta aparte
        en self.sa_size: la implementacion lo usa para todos los ordenes >= 3.)"""
        ent = self.meta["entries"]
        cap = {"WB-5": 5, "KN-5": 5, "EXCL-5": 5, "WB-12": 12, "KN-12": 12,
               "WB-20": 20, "KN-20": 20}.get(self.variant, DMAX)
        k = min(cap, DMAX)
        return int(ent[1:k + 1].sum()) + self.V

    @property
    def sa_size(self):
        return int(self.sa.shape[0])

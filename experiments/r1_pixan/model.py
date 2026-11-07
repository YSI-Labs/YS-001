"""Pixan: estado recurrente FIJO (pesos aleatorios, nunca entrenados) + SimHash + tablas de conteo.

Estado: K integradores con fugas de constante de tiempo tau_k (geometricas, 1..256 por defecto):
    h_k(t) = sum_{j<t} (1-a_k) a_k^(t-1-j) E[x_j],   a_k = exp(-1/tau_k),   E = embedding aleatorio fijo.
h(t) resume toda la historia (con memoria exponencial) en un vector continuo de dim D = K*d.
Cuantizacion: T tablas independientes; cada una tiene R=b hiperplanos aleatorios (SimHash) sobre h(t)
(estandarizado con estadisticas de train). bucket_t(t) = signos de los b hiperplanos.
Entrenamiento = proyeccion + conteo de (bucket_t, x_t) en cada tabla (cero gradientes).
Prediccion: por tabla, estimador Witten-Bell sobre el bucket (fallback = unigrama suavizado),
mezcla de las T tablas con pesos = masa de conteo N_t del bucket (parameter-free).

Todo lo que es "random" es fijo y se deriva de la semilla; no hay learning rate.
"""
import os
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import numpy as np
import numba as nb


@nb.njit(cache=False)
def states_at(x, E2, a, ps):
    """ps: posiciones ORDENADAS ascendentes. out[j] = h tras consumir x[0..ps[j]-1] (solo pasado)."""
    D = E2.shape[1]
    m = ps.shape[0]
    out = np.empty((m, D), np.float32)
    if m == 0:
        return out
    h = np.zeros(D, np.float64)
    maxp = ps[m - 1]
    j = 0
    for i in range(maxp + 1):
        while j < m and ps[j] == i:
            for k in range(D):
                out[j, k] = h[k]
            j += 1
        if i < maxp:
            c = x[i]
            for k in range(D):
                h[k] = a[k] * h[k] + (1.0 - a[k]) * E2[c, k]
    return out


class _Table:
    """Tabla de conteos dispersa: pares (bucket, char) -> conteo. Estimador Witten-Bell por bucket."""

    def __init__(self, keys, NB, V):
        # keys = bucket*V + char (int32/int64), NB = numero de buckets (incluye dummy)
        uk, cnt = np.unique(keys, return_counts=True)
        self.V = V
        self.uk = uk.astype(np.int64 if uk.dtype == np.int64 else np.int64)
        self.cnt = cnt.astype(np.float32)
        bk = uk // V
        self.tot = np.bincount(bk, weights=cnt, minlength=NB).astype(np.float32)
        self.typ = np.bincount(bk, minlength=NB).astype(np.float32)
        self.nnz = len(uk)

    def probs(self, bucket, p_lo):
        """bucket: int64 [m] (valores < NB). Devuelve (p [m,V] normalizada, N [m] masa de conteo)."""
        V = self.V
        q = bucket[:, None] * V + np.arange(V, dtype=np.int64)[None, :]
        idx = np.searchsorted(self.uk, q)
        idx = np.minimum(idx, len(self.uk) - 1)
        hit = self.uk[idx] == q
        c = np.where(hit, self.cnt[idx], 0.0).astype(np.float32)
        N = self.tot[bucket]
        U = self.typ[bucket]
        den = (N + U)
        p = (c + U[:, None] * p_lo[None, :]) / np.maximum(den, 1.0)[:, None]
        p = np.where(den[:, None] > 0, p, p_lo[None, :])
        return p.astype(np.float32), N


class _Base:
    def _unigram(self, train, V):
        cnt = np.bincount(train, minlength=V).astype(np.float64)
        p = (cnt + 1.0) / (cnt.sum() + V)  # Laplace: unica constante de suavizado del fallback
        return p.astype(np.float32)

    def _mix(self, tables_probs, Ns, p_lo, m, V):
        num = np.zeros((m, V), np.float32)
        den = np.zeros(m, np.float32)
        for p, N in zip(tables_probs, Ns):
            num += N[:, None] * p
            den += N
        out = np.where(den[:, None] > 0, num / np.maximum(den, 1e-30)[:, None], p_lo[None, :])
        out = out / out.sum(1, keepdims=True)
        return out.astype(np.float32)


class Pixan(_Base):
    def __init__(self, T=8, b=16, taus=None, d=4, seed=0, name=None):
        self.T, self.b, self.d, self.seed = T, b, d, seed
        assert b <= 20
        self.taus = np.asarray(taus if taus is not None else 2.0 ** np.arange(0, 9), dtype=np.float64)
        self.K = len(self.taus)
        self.name = name or f"Pixan-T{T}-b{b}-K{self.K}-d{d}"

    def fit(self, train, V, chunk=65536):
        self.V = V
        n = len(train)
        rng = np.random.default_rng(self.seed)
        D = self.K * self.d
        self.E2 = rng.standard_normal((V, D)).astype(np.float32)
        self.a = np.repeat(np.exp(-1.0 / self.taus), self.d).astype(np.float64)
        x = np.ascontiguousarray(train, dtype=np.uint8)
        H = states_at(x, self.E2, self.a, np.arange(n, dtype=np.int64))  # estado ANTES de cada x[i]
        self.mu = H.mean(0, dtype=np.float64).astype(np.float32)
        self.sd = (H.std(0, dtype=np.float64) + 1e-8).astype(np.float32)
        H -= self.mu
        H /= self.sd
        self.W = rng.standard_normal((D, self.T * self.b)).astype(np.float32)  # hiperplanos SimHash
        self.pow2 = (1 << np.arange(self.b, dtype=np.int64))
        keys = np.empty((self.T, n), dtype=np.int64 if (1 << self.b) * V >= 2**31 else np.int32)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            bk = self._buckets_from_std(H[s:e])  # [c, T]
            keys[:, s:e] = (bk * V + x[s:e, None].astype(np.int64)).T
        del H
        self.tables = []
        for t in range(self.T):
            self.tables.append(_Table(keys[t], 1 << self.b, V))
        del keys
        self.p_lo = self._unigram(x, V)

    def _buckets_from_std(self, Hs):
        c = Hs.shape[0]
        P = Hs @ self.W  # [c, T*b]
        bits = (P.reshape(c, self.T, self.b) > 0).astype(np.int64)
        return bits @ self.pow2  # [c, T]

    def predict(self, ids, pos):
        pos = np.asarray(pos, dtype=np.int64)
        order = np.argsort(pos, kind="stable")
        ps = pos[order]
        x = np.ascontiguousarray(ids, dtype=np.uint8)
        H = states_at(x, self.E2, self.a, ps)
        H -= self.mu
        H /= self.sd
        B = self._buckets_from_std(H)  # [m, T]
        m = len(ps)
        probs, Ns = [], []
        for t in range(self.T):
            p, N = self.tables[t].probs(B[:, t], self.p_lo)
            probs.append(p)
            Ns.append(N)
        out_sorted = self._mix(probs, Ns, self.p_lo, m, self.V)
        out = np.empty_like(out_sorted)
        out[order] = out_sorted
        return out

    def n_params(self):
        return int(sum(t.nnz for t in self.tables[: self.T]))


class ExactCtx(_Base):
    """Ablacion SIN SimHash: la clave es exactamente los ultimos k caracteres (misma estimacion WB-leaf)."""

    def __init__(self, k=4):
        self.k = k
        self.name = f"ExactCtx-k{k}"

    def _ctx(self, x, pos):
        V = self.V
        c = np.zeros(len(pos), dtype=np.int64)
        for j in range(1, self.k + 1):
            valid = pos - j >= 0
            v = np.zeros(len(pos), dtype=np.int64)
            v[valid] = x[pos[valid] - j]
            c += v * (V ** (j - 1))
        return c

    def fit(self, train, V):
        self.V = V
        n = len(train)
        x = np.ascontiguousarray(train, dtype=np.uint8)
        ctx = self._ctx(x.astype(np.int64), np.arange(n, dtype=np.int64))
        uc, inv = np.unique(ctx, return_inverse=True)
        self.uc = uc
        NB = len(uc)
        keys = inv.astype(np.int64) * V + x.astype(np.int64)
        self.tab = _Table(keys, NB + 1, V)  # bucket NB = dummy sin conteos
        self.p_lo = self._unigram(x, V)

    def predict(self, ids, pos):
        pos = np.asarray(pos, dtype=np.int64)
        x = np.asarray(ids).astype(np.int64)
        ctx = self._ctx(x, pos)
        j = np.minimum(np.searchsorted(self.uc, ctx), len(self.uc) - 1)
        found = self.uc[j] == ctx
        bucket = np.where(found, j, len(self.uc)).astype(np.int64)
        p, N = self.tab.probs(bucket, self.p_lo)
        return self._mix([p], [N], self.p_lo, len(pos), self.V)

    def n_params(self):
        return int(self.tab.nnz)

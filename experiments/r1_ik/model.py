"""IK: aprendizaje espectral de estado latente por solucion cerrada (sin gradientes).

Pipeline (todo conteo + algebra lineal):
  1. Historia h_t: vector disperso de dimension D. Para cada longitud n=1..L (L=12) se hashea el
     n-grama sufijo x[t-n..t-1] en un bucket de [0, D). nnz = L por fila.
  2. Futuro f_t: one-hot de x[t] (V dims) + one-hot del bigrama futuro x[t]x[t+1] (V*V dims). nnz = 2.
  3. C = E[h f^T] (D x (V+V^2)) se acumula por conteo (producto de matrices dispersas).
     SVD aleatorizada (Halko, 2 iteraciones de potencia) de rango r: C ~= U S V^T.
     Estado: u_t = U^T h_t en R^r (proyeccion lineal de la historia).
  4. Lectura en forma cerrada: ridge sobre [u_t, 1] -> one-hot(x[t]), con Gram acumulado.
     Salida lineal -> clip a >=0 + eps, normalizar (la masa negativa se mide y se documenta).
  5. Opcional: mezcla p = a*p_spec + (1-a)*p_WB(orden 5), con a elegido en dev.

Seleccion (lambda, a) SOLO con dev = ultimo 5% del train (nunca val). El modelo final se entrena
con el 95% inicial del train (dev queda fuera; decision documentada en REPORT.md).
"""
import os
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import sys
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
from baseline_wb import WittenBell  # n-gram de referencia (solo para la mezcla opcional)

L = 12                                   # ventana de historia (n-gramas sufijo de longitud 1..L)
B_ROLL = np.uint64(1000003)              # base del hash polinomico (constante, no ajustada)
C_SEED = np.uint64(0x9E3779B97F4A7C15)   # constantes de mezcla splitmix (no ajustadas)
C_MIX = np.uint64(0xBF58476D1CE4E5B9)
_NSEED = [np.uint64((n * int(C_SEED)) % (1 << 64)) for n in range(L + 1)]  # n*C_SEED mod 2^64 (precalculado)
CH = 500_000                             # tamano de chunk para la acumulacion por conteo
CH_REG = 50_000                          # chunk para el Gram de la regresion


def hist_buckets(ids, pos, Dbits):
    """(m, L) indices de bucket. Columna n-1 = hash del sufijo de longitud n que termina en pos-1."""
    pos = np.asarray(pos, np.int64)
    H = np.zeros(len(pos), np.uint64)
    out = np.empty((len(pos), L), np.int64)
    shift = np.uint64(64 - Dbits)
    for n in range(1, L + 1):
        idx = pos - n
        c = np.where(idx >= 0, ids[np.maximum(idx, 0)].astype(np.uint64) + np.uint64(1), np.uint64(0))
        H = H * B_ROLL + c
        h = (H + _NSEED[n]) * C_MIX
        h ^= h >> np.uint64(29)
        out[:, n - 1] = (h >> shift).astype(np.int64)
    return out


def hist_csr(ids, pos, Dbits):
    b = hist_buckets(ids, pos, Dbits)
    m = b.shape[0]
    indptr = np.arange(0, m * L + 1, L, dtype=np.int64)
    M = sp.csr_matrix((np.ones(m * L, np.float32), b.ravel(), indptr), shape=(m, 1 << Dbits))
    M.sum_duplicates()
    return M


def fut_csr(ids, ts, V):
    a = ids[ts].astype(np.int64)
    b = ids[ts + 1].astype(np.int64)
    m = len(ts)
    idx = np.stack([a, V + a * V + b], 1).ravel()
    indptr = np.arange(0, 2 * m + 1, 2, dtype=np.int64)
    return sp.csr_matrix((np.ones(2 * m, np.float32), idx, indptr), shape=(m, V + V * V))


def rsvd(A, k, p=10, q=2, seed=0):
    """SVD aleatorizada (Halko et al.). Devuelve U (n_rows x k), S (k), V (n_cols x k)."""
    rng = np.random.default_rng(seed)
    Om = rng.standard_normal((A.shape[1], k + p))
    Q, _ = np.linalg.qr(A @ Om)
    for _ in range(q):
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    Bt = A.T @ Q                                   # (n_cols x k+p) = B^T con B = Q^T A
    X, S, Yt = np.linalg.svd(Bt, full_matrices=False)
    U = Q @ Yt.T
    return U[:, :k], S[:k], X[:, :k]


def _clipnorm(raw, eps):
    p = np.maximum(raw, 0.0) + eps
    return p / p.sum(1, keepdims=True)


class IKSpectral:
    def __init__(self, r=256, Dbits=14, mix="dev", lam_grid=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                 alpha_grid=tuple(np.round(np.linspace(0, 1, 11), 2)), wb_order=5,
                 n_reg=500_000, n_dev_eval=20_000, dev_frac=0.05, seed=0, eps=1e-6, proj="svd"):
        self.proj = proj
        self.r, self.Dbits, self.mix = r, Dbits, mix
        self.lam_grid, self.alpha_grid = lam_grid, alpha_grid
        self.wb_order, self.n_reg, self.n_dev_eval = wb_order, n_reg, n_dev_eval
        self.dev_frac, self.seed, self.eps = dev_frac, seed, eps
        self.D = 1 << Dbits
        self.name = f"IK-spec-r{r}-D2^{Dbits}-" + ("mix" if mix == "dev" else "pure") + ("" if proj == "svd" else "-" + proj)
        self.info = {}
        self.alpha = 1.0
        self.wb = None

    def n_params(self):
        V = self.V
        n = self.D * self.r + (self.r + 1) * V
        if self.alpha < 1.0 and self.wb is not None:
            n += self.wb.n_params()
        return int(n)

    def fit(self, train, V):
        self.V = V
        r, D = self.r, self.D
        N = len(train)
        Ntr = N - int(N * self.dev_frac)
        self.info.update(N_fit=Ntr, N_dev=N - Ntr, n_reg=min(self.n_reg, Ntr))

        # 1) conteo: C = sum_t h_t f_t^T sobre el tramo de fit
        ts_all = np.arange(1, Ntr - 1)
        C = None
        for s in range(0, len(ts_all), CH):
            ts = ts_all[s:s + CH]
            Cc = (hist_csr(train, ts, self.Dbits).T @ fut_csr(train, ts, V)).tocsr()
            C = Cc if C is None else C + Cc
        C = C.astype(np.float64)
        C.sum_duplicates()
        self.info["C_nnz"] = int(C.nnz)

        # 2) SVD de rango r -> proyeccion de historia
        U, S, _ = rsvd(C, r, seed=self.seed)
        del C
        if self.proj == "random":  # control: proyeccion gaussiana aleatoria, mismo pipeline
            U = np.random.default_rng(self.seed + 1).standard_normal((D, r)) / np.sqrt(D)
            S = np.ones(r)
        self.U = U.astype(np.float32)
        self.info["sv_top"] = float(S[0])
        self.info["sv_r"] = float(S[-1])

        # 3) Gram de la regresion de lectura sobre un subconjunto uniforme del fit
        reg_ts = np.linspace(1, Ntr - 2, min(self.n_reg, Ntr - 2)).astype(np.int64)
        G = np.zeros((r + 1, r + 1))
        Bm = np.zeros((r + 1, V))
        for s in range(0, len(reg_ts), CH_REG):
            ts = reg_ts[s:s + CH_REG]
            Z = np.ones((len(ts), r + 1))
            Z[:, :r] = hist_csr(train, ts, self.Dbits) @ self.U
            Y = np.zeros((len(ts), V))
            Y[np.arange(len(ts)), train[ts].astype(np.int64)] = 1.0
            G += Z.T @ Z
            Bm += Z.T @ Y
        scale = np.trace(G[:r, :r]) / r

        # 4) dev: elegir lambda (y alpha de mezcla) sin mirar val
        self.wb = WittenBell(self.wb_order)
        self.wb.fit(train[:Ntr], V)
        dev_ts = np.linspace(Ntr, N - 1, self.n_dev_eval).astype(np.int64)
        Zd = np.ones((len(dev_ts), r + 1))
        Zd[:, :r] = hist_csr(train, dev_ts, self.Dbits) @ self.U
        yd = train[dev_ts].astype(np.int64)
        pwb = self.wb.predict(train, dev_ts).astype(np.float64)
        pw_true = pwb[np.arange(len(yd)), yd]

        alphas = (1.0,) if self.mix != "dev" else tuple(self.alpha_grid)
        table = []
        best = None
        for lam in self.lam_grid:
            W = np.linalg.solve(G + lam * scale * np.eye(r + 1), Bm)
            raw = Zd @ W
            neg = np.minimum(raw, 0.0).sum(1)           # masa negativa por fila (prob. units)
            ps = _clipnorm(raw, self.eps)
            ps_true = ps[np.arange(len(yd)), yd]
            for a in alphas:
                if a == 1.0:
                    bpc = float(-np.log2(np.maximum(ps_true, 1e-12)).mean())
                else:
                    pt = a * ps_true + (1 - a) * pw_true
                    bpc = float(-np.log2(np.maximum(pt, 1e-12)).mean())
                table.append(dict(lam=float(lam), alpha=float(a), dev_bpc=bpc,
                                  neg_mass_mean=float(-neg.mean()), frac_rows_neg=float((neg < 0).mean()),
                                  raw_sum_mean=float(raw.sum(1).mean())))
                if best is None or bpc < best[0]:
                    best = (bpc, lam, a)
        self.info["dev_table"] = table
        self.info["dev_wb_bpc"] = float(-np.log2(np.maximum(pw_true, 1e-12)).mean())
        _, lam_b, a_b = best
        self.lam, self.alpha = float(lam_b), float(a_b)
        self.info.update(lam_selected=self.lam, alpha_selected=self.alpha, dev_bpc_selected=best[0])

        # 5) pesos finales de lectura con lambda elegido (entrenados sobre el fit, dev fuera)
        self.W = np.linalg.solve(G + self.lam * scale * np.eye(r + 1), Bm)
        if self.alpha >= 1.0:
            self.wb = None
        return self

    def predict(self, ids, pos):
        pos = np.asarray(pos, np.int64)
        out = np.empty((len(pos), self.V), np.float32)
        for s in range(0, len(pos), 4096):
            pp = pos[s:s + 4096]
            u = (hist_csr(ids, pp, self.Dbits) @ self.U).astype(np.float64)
            Z = np.concatenate([u, np.ones((len(pp), 1))], 1)
            ps = _clipnorm(Z @ self.W, self.eps)
            if self.alpha < 1.0:
                pw = self.wb.predict(ids, pp).astype(np.float64)
                p = self.alpha * ps + (1 - self.alpha) * pw
            else:
                p = ps
            out[s:s + len(pp)] = (p / p.sum(1, keepdims=True)).astype(np.float32)
        return out

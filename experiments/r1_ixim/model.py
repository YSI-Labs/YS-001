"""Ixim: capa oculta FIJA ALEATORIA + salida lineal resuelta en forma cerrada (ridge) + calibracion
a distribucion. Cero backprop, cero learning rate.

Pipeline (modo "random"):
  contexto x = ids[pos-L:pos]
  Z = sum_{j=1..L} E[j, x_j]           E: (L, V+1, d0) gaussiana fija (V = pad de inicio)  -> R^d0
  H = ReLU(Z P)                        P: (d0, D) gaussiana fija                            -> R^D
  yhat = [H, 1] W                      W: solucion cerrada de ridge sobre el Gram de train
  p = calibracion(yhat)                softmax(s*yhat) | clip+normaliza | softmax(s*yhat+b) (Newton)

Modo "ngram" (control): mismos ejes, pero las features son indicadores one-hot de los n-gramas
suffix de orden 1..K mas frecuentes en train (presupuesto D repartido de forma greedy por orden).

Particion (solo train, nunca val): train[:95%] para Gram y ridge; train[95%:] = dev para
calibracion y cross-check de lambda. Val SOLO lo toca harness.evaluate al final.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import numba as nb

CHUNK = 8192          # filas por bloque en el Gram (RAM acotada)
DEV_FRAC = 0.05       # ultimo 5% de train = dev
DEV_SAMPLE = 40_000   # filas de dev usadas (mitad A para ajustar calibracion, mitad B para elegirla)
T0 = 32               # primera posicion de entrenamiento (contexto minimo completo)


@nb.njit(parallel=True, fastmath=True)
def _embed_sum(ids, pos, E, pad):
    L = E.shape[0]
    d0 = E.shape[2]
    n = pos.shape[0]
    Z = np.zeros((n, d0), np.float32)
    for i in nb.prange(n):
        p = pos[i]
        for j in range(L):
            t = p - 1 - j
            c = pad
            if t >= 0:
                c = np.int64(ids[t])
            for k in range(d0):
                Z[i, k] += E[j, c, k]
    return Z


def _ngram_keys(ids, pos, K, V):
    """keys[k-1, i] = codigo base-V del sufijo de longitud k que termina en pos[i]-1."""
    pos = np.asarray(pos, np.int64)
    assert pos.min() >= K, "pos debe ser >= K"
    keys = np.zeros((K, len(pos)), np.int64)
    acc = np.zeros(len(pos), np.int64)
    mult = 1
    for k in range(1, K + 1):
        acc = acc + ids[pos - k].astype(np.int64) * mult
        mult *= V
        keys[k - 1] = acc
    return keys


def _nll_bits(P, y):
    return float(-np.log2(np.maximum(P[np.arange(len(y)), y], 1e-300)).mean())


def _softmax_scaled(Yh, s):
    Z = s * Yh
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def _clip_norm(Yh, eps):
    P = np.maximum(Yh, eps)
    return P / P.sum(1, keepdims=True)


def _newton_calib(Yh, y, s0, iters=8):
    """Newton completo sobre theta=(s,b) de p=softmax(s*yhat+b). Problema convexo (logits lineales
    en theta). Pseudoinversa para la direccion nula de b (invariancia a sumar constante)."""
    n, V = Yh.shape
    Y1 = np.zeros((n, V), np.float64)
    Y1[np.arange(n), y] = 1.0
    s, b = float(s0), np.zeros(V)
    for _ in range(iters):
        Z = s * Yh + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        R = P - Y1
        g = np.concatenate([[(R * Yh).sum()], R.sum(0)])
        py = (P * Yh).sum(1, keepdims=True)
        Hbb = np.diag(P.sum(0)) - P.T @ P
        Hbs = (P * Yh - P * py).sum(0)
        Hss = (P * Yh ** 2).sum() - (py ** 2).sum()
        H = np.empty((V + 1, V + 1))
        H[0, 0] = Hss
        H[0, 1:] = Hbs
        H[1:, 0] = Hbs
        H[1:, 1:] = Hbb
        step = np.linalg.pinv(H, rcond=1e-10) @ g
        s -= step[0]
        b -= step[1:]
    return s, b


class Ixim:
    def __init__(self, mode="random", L=32, D=2048, d0=256, K=4, seed=0):
        assert mode in ("random", "ngram")
        self.mode, self.L, self.D, self.d0, self.K, self.seed = mode, L, D, d0, K, seed
        self.name = (f"Ixim-random-L{L}-D{D}-d0{d0}" if mode == "random"
                     else f"Ixim-ngram{K}-D{D}")
        self.calib = "softmax"  # se fija tras fit (eleccion por dev); se puede sobreescribir para reportar

    # ------------------------------------------------------------------ features
    def _features(self, ids, pos):
        n = len(pos)
        if self.mode == "random":
            Z = _embed_sum(ids, pos, self.E, self.V)
            H = Z @ self.P
            np.maximum(H, 0, out=H)
            X = np.empty((n, self.D + 1), np.float32)
            X[:, :self.D] = H
        else:
            keys = _ngram_keys(ids, pos, self.K, self.V)
            X = np.zeros((n, self.Dx + 1), np.float32)
            rows = np.arange(n)
            for k in range(self.K):
                tkeys, tcols = self.tables[k]
                idx = np.minimum(np.searchsorted(tkeys, keys[k]), len(tkeys) - 1)
                hit = tkeys[idx] == keys[k]
                X[rows[hit], tcols[idx[hit]]] = 1.0
        X[:, -1] = 1.0
        return X

    def _build_ngram_tables(self, ids, V):
        pos = np.arange(T0, self.dev_start, dtype=np.int64)
        keys = _ngram_keys(ids, pos, self.K, V)
        budget, off, self.tables = self.D, 0, []
        for k in range(1, self.K + 1):
            u, c = np.unique(keys[k - 1], return_counts=True)
            m_k = int(min(len(u), budget // (self.K - k + 1)))
            top = np.argsort(-c, kind="stable")[:m_k]
            sel = u[top]
            order = np.argsort(sel)
            self.tables.append((sel[order], off + order))
            budget -= m_k
            off += m_k
        self.Dx = off

    # ------------------------------------------------------------------ fit
    def fit(self, train, V):
        self.V = V
        N = len(train)
        self.dev_start = int(N * (1 - DEV_FRAC))
        rng = np.random.default_rng(self.seed)
        if self.mode == "random":
            self.E = (rng.standard_normal((self.L, V + 1, self.d0)) / np.sqrt(self.d0)).astype(np.float32)
            self.P = (rng.standard_normal((self.d0, self.D)) / np.sqrt(self.d0)).astype(np.float32)
            self.Dx = self.D
        else:
            self._build_ngram_tables(train, V)
        Dt = self.Dx + 1
        self.Dt = Dt

        # 1) Gram conjunto [X, Y] sobre train[:dev_start] por bloques
        pos_tr = np.arange(T0, self.dev_start, dtype=np.int64)
        M = Dt + V
        G = np.zeros((M, M), np.float64)
        for s in range(0, len(pos_tr), CHUNK):
            p = pos_tr[s:s + CHUNK]
            X = self._features(train, p)
            Y = np.zeros((len(p), V), np.float32)
            Y[np.arange(len(p)), train[p]] = 1.0
            A = np.hstack([X, Y])
            G += (A.T @ A).astype(np.float64)
        n_tr = float(len(pos_tr))
        Gxx, Gxy = G[:Dt, :Dt], G[:Dt, Dt:]
        trYY = float(np.trace(G[Dt:, Dt:]))

        # 2) lambda por GCV en forma cerrada (una autodescomposicion sirve para toda la rejilla)
        sv, Q = np.linalg.eigh(Gxx)
        sv = np.maximum(sv, 0.0)
        Bt = Q.T @ Gxy
        scale = sv.mean()
        lam_grid = np.logspace(-8, 2, 41) * scale  # rejilla relativa a la traza media (sin unidades)
        gcv, rss_l, trh_l = [], [], []
        for lam in lam_grid:
            f = 1.0 / (sv + lam)
            Wt = f[:, None] * Bt
            rss = trYY - 2.0 * np.sum(Wt * Bt) + np.sum(sv[:, None] * Wt ** 2)
            trh = float(np.sum(sv * f))
            gcv.append(n_tr * rss / (n_tr - trh) ** 2)
            rss_l.append(rss)
            trh_l.append(trh)
        k = int(np.argmin(gcv))
        self.lam = float(lam_grid[k])
        self.lam_rel = float(lam_grid[k] / scale)
        self.gcv_curve = (np.array(gcv) / n_tr).tolist()
        self.W = Q @ ((1.0 / (sv + self.lam))[:, None] * Bt)  # (Dt, V) solucion de ridge
        self.W32 = self.W.astype(np.float32)

        # 3) Dev: features de una muestra uniforme; calibracion ajustada en A, elegida por NLL en B
        pos_dev = np.linspace(self.dev_start, N - 1, DEV_SAMPLE).astype(np.int64)
        yd = train[pos_dev].astype(np.int64)
        Xd = self._features(train, pos_dev)
        Yh = (Xd @ self.W32).astype(np.float64)
        h = len(pos_dev) // 2
        YA, yA, YB, yB = Yh[:h], yd[:h], Yh[h:], yd[h:]
        t0 = time_now()
        # (a) softmax(s*yhat), s en rejilla log
        s_grid = np.logspace(-1, 3, 41)
        nlla = [_nll_bits(_softmax_scaled(YA, s), yA) for s in s_grid]
        sa = float(s_grid[int(np.argmin(nlla))])
        # (b) clip + normalizar, eps en rejilla log
        e_grid = np.logspace(-8, 0, 33)
        nllb = [_nll_bits(_clip_norm(YA, e), yA) for e in e_grid]
        eb = float(e_grid[int(np.argmin(nllb))])
        # (c) Newton sobre (s, b) partiendo de (a)
        sc, bc = _newton_calib(YA, yA, sa, iters=8)
        self.cal_params = {"softmax": sa, "clip": eb, "newton": (sc, bc.copy())}
        self.calib_fit_s = time_now() - t0
        self.dev_nll_B = {
            "softmax": _nll_bits(_softmax_scaled(YB, sa), yB),
            "clip": _nll_bits(_clip_norm(YB, eb), yB),
            "newton": _nll_bits(_softmax_scaled_b(YB, sc, bc), yB),
        }
        self.dev_nll_A = {
            "softmax": _nll_bits(_softmax_scaled(YA, sa), yA),
            "clip": _nll_bits(_clip_norm(YA, eb), yA),
            "newton": _nll_bits(_softmax_scaled_b(YA, sc, bc), yA),
        }
        # Cross-check de lambda en dev: MSE one-hot (no usa val)
        Y1 = np.zeros((len(yd), V))
        Y1[np.arange(len(yd)), yd] = 1
        self.dev_mse_curve = []
        for lam in lam_grid:
            Wl = (Q @ ((1.0 / (sv + lam))[:, None] * Bt)).astype(np.float32)
            self.dev_mse_curve.append(float(((Xd @ Wl - Y1.astype(np.float32)) ** 2).mean()))
        self.dev_lam_argmin_rel = float(lam_grid[int(np.argmin(self.dev_mse_curve))] / scale)
        self.calib = min(self.dev_nll_B, key=self.dev_nll_B.get)  # eleccion por dev
        self.fit_info = dict(n_train_rows=int(n_tr), Dx=int(self.Dx), lam_rel_gcv=self.lam_rel,
                             lam_rel_dev_mse_argmin=self.dev_lam_argmin_rel,
                             cal_params={k_: (v if not isinstance(v, tuple) else [float(v[0]), v[1].tolist()])
                                         for k_, v in self.cal_params.items()},
                             dev_nll_A=self.dev_nll_A, dev_nll_B=self.dev_nll_B,
                             calib_chosen_by_dev_B=self.calib)

    # ------------------------------------------------------------------ predict
    def _apply_calib(self, Yh, mode):
        if mode == "softmax":
            return _softmax_scaled(Yh, self.cal_params["softmax"])
        if mode == "clip":
            return _clip_norm(Yh, self.cal_params["clip"])
        if mode == "newton":
            s, b = self.cal_params["newton"]
            return _softmax_scaled_b(Yh, s, b)
        raise ValueError(mode)

    def predict(self, ids, pos):
        pos = np.asarray(pos, np.int64)
        Yh = (self._features(ids, pos) @ self.W32).astype(np.float64)
        P = self._apply_calib(Yh, self.calib)
        P = np.maximum(P, 1e-12)  # piso numerico (contrato: p > 0)
        P /= P.sum(1, keepdims=True)
        return P.astype(np.float32)

    def n_params(self):
        # parametros aprendidos: W (Dt x V) + calibracion Newton (V+1). Las matrices E, P son fijas (no entrenadas).
        return int(self.Dt * self.V + self.V + 1)

    def n_fixed_random(self):
        if self.mode == "random":
            return int(self.L * (self.V + 1) * self.d0 + self.d0 * self.D)
        return 0


def _softmax_scaled_b(Yh, s, b):
    Z = s * Yh + b
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def time_now():
    import time
    return time.time()

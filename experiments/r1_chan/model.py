"""Chan: modelo de caracteres con computacion hiperdimensional (VSA), sin entrenamiento iterativo.

Hipervectores bipolares {-1,+1}^D. Cada caracter v tiene un hipervector fijo E[v] (aleatorio, semilla fija).
Contexto de orden k en la posicion t: K_k(t) = prod_{j=1..k} rho^j(E[x_{t-j}])  (ligadura posicional
por permutacion ciclica rho + producto bipolar = XOR en bits). Es el hash exacto de un k-grama.

Modos:
  "proto": un prototipo por caracter siguiente, M_c = sum_{t: x_t=c} C(t), con C(t) = sum_{k in orders} K_k(t)
           (superposicion de ordenes). Puntaje z_c = <C(t), M_c>/(D*N) ; p = softmax(beta*z). beta = unico
           hiperparametro, ajustado por busqueda 1-D en dev (ver sweep.py).
  "wb":    una memoria por orden: M^(k)_c = sum_{t: x_t=c} K_k(t). Conteo estimado del k-grama+c:
           s^(k)_c = <K_k(t), M^(k)_c>/D (= conteo exacto + ruido). Mezcla tipo Witten-Bell recursiva con esos
           conteos (sin beta). Umbral 0.5 y suelo U>=1 son las unicas constantes de la mezcla.

Entrenamiento = una pasada de sumas enteras (numba, serial). Sin gradientes, sin learning rate.
"""
import os
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np
from numba import njit


@njit(cache=True)
def _fit_kernel(x, start, end, R, K, mask, Mproto, Mord, do_proto, do_ord):
    D = R.shape[2]
    p = np.empty(D, np.int32)
    C = np.empty(D, np.int32)
    for t in range(start, end):
        c = x[t]
        for d in range(D):
            p[d] = 1
            C[d] = 0
        for k in range(1, K + 1):
            ch = x[t - k]
            for d in range(D):
                p[d] *= R[k, ch, d]
            if mask[k]:
                for d in range(D):
                    C[d] += p[d]
            if do_ord:
                for d in range(D):
                    Mord[k, c, d] += p[d]
        if do_proto:
            for d in range(D):
                Mproto[c, d] += C[d]


@njit(cache=True)
def _fit_wb_kernel(x, start, end, R, K, bkt, Mb):
    D = R.shape[2]
    p = np.empty(D, np.int32)
    for t in range(start, end):
        c = x[t]
        b = bkt[t]
        for d in range(D):
            p[d] = 1
        for k in range(1, K + 1):
            ch = x[t - k]
            for d in range(D):
                p[d] *= R[k, ch, d]
            for d in range(D):
                Mb[k, b, c, d] += p[d]


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class ChanHDC:
    CHUNK = 1024

    def __init__(self, D=4096, K=8, mode="proto", orders=None, seed=0, beta=1.0, name=None, H=1, zthr=0.0):
        self.D, self.K, self.mode, self.seed = D, K, mode, seed
        self.H, self.zthr = H, zthr  # wb: particiones por hash de (x_{t-1},x_{t-2}) mod H; umbral de ruido en sigmas
        self.orders = list(range(1, K + 1)) if orders is None else list(orders)
        self.beta = beta
        self.name = name or f"Chan-{mode}-D{D}-K{K}" + ("" if orders is None else "-ord" + "".join(map(str, self.orders)))             + (f"-H{H}-z{zthr}" if mode == "wb" else "")

    def _codebook(self, V):
        rng = np.random.default_rng(self.seed)
        E = rng.choice(np.array([-1, 1], dtype=np.int8), size=(V, self.D))
        # R[k, v, d] = rho^k(E[v])[d] = E[v, (d-k) mod D]
        return np.stack([np.roll(E, k, axis=1) for k in range(self.K + 1)]).astype(np.int8)

    def fit(self, train, V):
        self.V = V
        D, K = self.D, self.K
        self.R = self._codebook(V)
        self.mask = np.zeros(K + 1, np.int64)
        self.mask[self.orders] = 1
        x = np.ascontiguousarray(train)
        self.uni = np.bincount(x, minlength=V).astype(np.float64)
        self.N = len(x) - K
        if self.mode == "proto":
            Mproto = np.zeros((V, D), np.int32)
            _fit_kernel(x, K, len(x), self.R, K, self.mask, Mproto, np.zeros((1, 1, 1), np.int32), True, False)
            self.Mf = (Mproto.astype(np.float32) / np.float32(D * self.N))  # (V, D)
            del Mproto
        else:
            # wb: bucket por posicion; Nb[b,c] = conteo exacto de c tras bucket b (sigma^2 del ruido = Nb/D)
            bkt = np.zeros(len(x), np.int64)
            bkt[K:] = (x[K - 1:-1].astype(np.int64) * V + x[K - 2:-2].astype(np.int64)) % self.H
            self.Nb = np.bincount(bkt[K:] * V + x[K:].astype(np.int64), minlength=self.H * V).reshape(self.H, V).astype(np.float64)
            Mb = np.zeros((K + 1, self.H, V, D), np.float32)
            _fit_wb_kernel(x, K, len(x), self.R, K, bkt, Mb)
            self.Mb = Mb
        return self

    def n_params(self):
        return self.V * self.D if self.mode == "proto" else self.K * self.H * self.V * self.D

    def raw_logits(self, ids, pos):
        """Proto: puntajes z (m,V) sin escalar por beta."""
        assert self.mode == "proto"
        pos = np.asarray(pos, np.int64)
        out = []
        for s in range(0, len(pos), self.CHUNK):
            pp = pos[s:s + self.CHUNK]
            C = np.zeros((len(pp), self.D), np.float32)
            p = np.ones((len(pp), self.D), np.float32)
            for k in range(1, self.K + 1):
                p *= self.R[k][ids[pp - k]]
                if self.mask[k]:
                    C += p
            out.append(C @ self.Mf.T)
        return np.concatenate(out)

    def predict(self, ids, pos):
        pos = np.asarray(pos, np.int64)
        if self.mode == "proto":
            return _softmax(self.beta * self.raw_logits(ids, pos)).astype(np.float32)
        return self._predict_wb(ids, pos).astype(np.float32)

    def _predict_wb(self, ids, pos):
        D, K, V = self.D, self.K, self.V
        out = []
        for s in range(0, len(pos), self.CHUNK):
            pp = pos[s:s + self.CHUNK]
            m = len(pp)
            bq = (ids[pp - 1].astype(np.int64) * V + ids[pp - 2].astype(np.int64)) % self.H
            sig = np.sqrt(self.Nb[bq] / D)                         # (m,V) desv. tipica exacta del ruido
            p_prev = np.repeat(((self.uni + 1) / (self.uni.sum() + V))[None, :], m, 0)
            p = np.ones((m, D), np.float32)
            for k in range(1, K + 1):
                p *= self.R[k][ids[pp - k]]
                s_k = np.empty((m, V), np.float32)
                for b in np.unique(bq):
                    rows = np.nonzero(bq == b)[0]
                    s_k[rows] = (p[rows] @ self.Mb[k, b].T) / D    # conteos estimados con ruido
                cnt = np.maximum(s_k - self.zthr * sig, 0.0)
                T = cnt.sum(1, keepdims=True)
                U = np.maximum(1.0, (s_k >= 0.5).sum(1, keepdims=True))
                p_prev = (cnt + U * p_prev) / (T + U)
            out.append(p_prev / p_prev.sum(1, keepdims=True))
        return np.concatenate(out)

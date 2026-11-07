"""Mezcla geometrica (producto de expertos) con pesos GLOBALES por maxima verosimilitud.

p(x) proporcional a prod_k p_k(x)^{w_k}.  La log-verosimilitud
    F(w) = sum_i [ w . s_i  -  log Z_i(w) ],   s_ik = log p_k(x_i),   Z_i = sum_x prod_k p_k(x)^{w_k}
es CONCAVA en w (log-sum-exp convexa), asi que Newton con la Hessiana exacta (covarianza de los log-expertos bajo
q_i) y busqueda de linea Armijo converge sin learning rate ni hiperparametros de optimizacion libres
(constantes numericas: c=1e-4, reduccion 1/2, ridge 1e-6, tolerancia 1e-7).
Origen: experimento r2_mixer (Haiku), reimplementado y fusionado por el orquestador.
"""
import numpy as np


def poe_global(LP, xs, iters=40, tol=1e-7, verbose=False):
    """LP: [K, m, V] log-probabilidades (float32) de cada experto; xs: [m] simbolo verdadero. Devuelve w [K]."""
    K, m, V = LP.shape
    ar = np.arange(m)
    s = np.stack([LP[k, ar, xs] for k in range(K)], axis=1).astype(np.float64)   # [m,K]
    S = s.sum(0)

    def logZ(w):
        L = np.zeros((m, V), dtype=np.float32)
        for k in range(K):
            L += np.float32(w[k]) * LP[k]
        mx = L.max(1)
        return L, mx + np.log(np.exp(L - mx[:, None], dtype=np.float32).sum(1, dtype=np.float64))

    def F(w):
        _, lse = logZ(w)
        return float(S @ w - lse.sum())

    w = np.full(K, 1.0 / K)
    f0 = F(w)
    for it in range(iters):
        L, lse = logZ(w)
        q = np.exp(L - lse[:, None].astype(np.float32), dtype=np.float32)               # [m,V]
        E = np.stack([(q * LP[k]).sum(1, dtype=np.float64) for k in range(K)], axis=1)   # [m,K]
        grad = S - E.sum(0)
        C = np.zeros((K, K))
        for j in range(K):
            qj = q * LP[j]
            for k in range(j, K):
                C[j, k] = float((qj * LP[k]).sum(dtype=np.float64)) - float((E[:, j] * E[:, k]).sum())
                C[k, j] = C[j, k]
        D = np.linalg.solve(C + 1e-6 * np.eye(K), grad)
        slope = float(grad @ D)
        alpha = 1.0
        for _ in range(12):
            f1 = F(w + alpha * D)
            if f1 >= f0 + 1e-4 * alpha * slope:
                break
            alpha *= 0.5
        else:
            break
        w = w + alpha * D
        if verbose:
            print(f"  poe it={it} F={f1/m:.5f} nats/char  w={np.round(w, 3).tolist()}", flush=True)
        if abs(f1 - f0) < tol * m:
            f0 = f1
            break
        f0 = f1
    return w


def poe_combine(LPs, w):
    """LPs: lista de K arrays [m,V] de log-probabilidades; devuelve probabilidades normalizadas [m,V] float32."""
    L = np.zeros(LPs[0].shape, dtype=np.float64)
    for k, lp in enumerate(LPs):
        L += w[k] * lp
    L -= L.max(1, keepdims=True)
    p = np.exp(L)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)

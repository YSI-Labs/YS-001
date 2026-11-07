"""Copia de ixchel/mixing.py (r4_codeatoms). UNICO cambio: el paso E = sum_v q LP se calcula por bloques de filas
(antes: LP.astype(float64) completo, ~2 GB con 18 expertos). Misma matematica; verificado en test_mixing_chunk.py."""
import numpy as np

CH = 20_000  # filas (posiciones) por bloque para acotar memoria


def poe_global(LP, xs, iters=40, tol=1e-7, verbose=False):
    """LP: [K, m, V] log-probabilidades (float32); xs: [m] simbolo verdadero. Devuelve w [K]."""
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
        E = np.zeros((m, K))
        for c0 in range(0, m, CH):
            sl = slice(c0, min(c0 + CH, m))
            E[sl] = np.einsum("mv,kmv->mk", q[sl].astype(np.float64), LP[:, sl, :].astype(np.float64), optimize=True)
        grad = S - E.sum(0)
        M = np.zeros((K, K))
        for c0 in range(0, m, CH):
            sl = slice(c0, min(c0 + CH, m))
            A = np.moveaxis(LP[:, sl, :], 0, -1).reshape(-1, K).astype(np.float64)     # [cV, K]
            qc = q[sl].reshape(-1, 1).astype(np.float64)
            M += (A * qc).T @ A
        C = M - E.T @ E
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

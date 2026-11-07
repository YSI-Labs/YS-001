"""R4 gate: mezcla geometrica (PoE) CON COMPUERTA por dominio/estilo, sin hiperparametros libres.

Modelo:  p_g(x) ~ exp( sum_k w[g,k] LP_k(x) )   con g = compuerta causal (ver detector).
Ajuste (todo sobre el tramo DEV interno del train; nunca val):
  1. MLE global w0 (Newton exacto, Hessiana = -C con C = covarianza de los LP bajo q).
  2. MLE por compuerta wh_g (Newton exacto sobre las filas de la compuerta).
  3. Fuerza del encogimiento por METODO DE MOMENTOS (Gauss jerarquico, Laplace):
        wh_g - w0 ~ N(0, tau^2 I + C_g^{-1})   =>   tau^2 = max(0, mean_g (||wh_g-w0||^2 - tr C_g^{-1}) / K)
     lambda = 1/tau^2  (lambda=inf => todas las compuertas colapsan a w0).
  4. MAP por compuerta: maximiza F_g(w) - lambda/2 ||w-w0||^2 con Newton exacto (empieza en w0).
Detector de compuerta (causal; solo historial reciente, 500 caracteres):
  rasgos = fraccion de cada clase de caracter (8) + fraccion de espacios de sangria en la ventana (1),
  escalados con la desviacion del tramo de entrenamiento; k-means++ con k in 1..8 elegido por BIC (X-means).
"""
import numpy as np
import numba as nb

CH = 20_000


# ------------------------------------------------------------ PoE: Newton con prior opcional
def _stats(LP, xs, w, S):
    """Devuelve F (sin penalizacion), gradiente sin penalizar, C (covarianza de LP bajo q) en w."""
    K, m, V = LP.shape
    L = np.zeros((m, V), dtype=np.float32)
    for k in range(K):
        L += np.float32(w[k]) * LP[k]
    mx = L.max(1)
    lse = mx + np.log(np.exp(L - mx[:, None], dtype=np.float32).sum(1, dtype=np.float64))
    F = float(S @ w - lse.sum())
    q = np.exp(L - lse[:, None].astype(np.float32), dtype=np.float32)
    del L
    E = np.zeros((m, K))
    for k in range(K):
        E[:, k] = (q * LP[k]).sum(1, dtype=np.float64)
    grad = S - E.sum(0)
    M = np.zeros((K, K))
    for c0 in range(0, m, CH):
        sl = slice(c0, min(c0 + CH, m))
        A = np.moveaxis(LP[:, sl, :], 0, -1).reshape(-1, K).astype(np.float64)
        qc = q[sl].reshape(-1, 1).astype(np.float64)
        M += (A * qc).T @ A
    C = M - E.T @ E
    return F, grad, C


def poe_map(LP, xs, w0, lam=0.0, iters=40, tol=1e-9):
    """Maximiza sum_i log p_w(x_i) - lam/2 ||w - w0||^2 (Newton exacto + Armijo).
    LP [K,m,V] float32 log-probs; xs [m]. Devuelve (w, C) con C la covarianza de LP en w (sin penalizar)."""
    K, m, V = LP.shape
    ar = np.arange(m)
    s = np.stack([LP[k, ar, xs] for k in range(K)], axis=1).astype(np.float64)
    S = s.sum(0)
    w = np.array(w0, dtype=np.float64) if w0 is not None else np.zeros(K)
    w0 = w.copy()
    F, grad, C = _stats(LP, xs, w, S)
    obj = F - 0.5 * lam * float(((w - w0) ** 2).sum())
    for it in range(iters):
        g = grad - lam * (w - w0)
        D = np.linalg.solve(C + (lam + 1e-6) * np.eye(K), g)
        slope = float(g @ D)
        if slope <= 0:
            break
        alpha = 1.0
        for _ in range(14):
            wn = w + alpha * D
            Fn, gn, Cn = _stats(LP, xs, wn, S)
            objn = Fn - 0.5 * lam * float(((wn - w0) ** 2).sum())
            if objn >= obj + 1e-4 * alpha * slope:
                break
            alpha *= 0.5
        else:
            break
        improved = objn - obj
        w, F, grad, C, obj = wn, Fn, gn, Cn, objn
        if improved < tol * max(m, 1):
            break
    return w, C


def fit_gated(LP, xs, gate, G, verbose=False):
    """Pesos W [G,K] por compuerta con encogimiento EB hacia el global. gate [m] en [0,G).
    Devuelve (W, info) ; info con tau2, lam, n por compuerta, ||w_g-w0||."""
    K, m, V = LP.shape
    w0, _ = poe_map(LP, xs, np.zeros(K), lam=0.0)
    n = np.bincount(gate, minlength=G)
    wh = np.tile(w0, (G, 1)); trinv = np.zeros(G)
    for g in range(G):
        if n[g] == 0:
            continue
        idx = np.flatnonzero(gate == g)
        sub = LP[:, idx, :]
        wg, Cg = poe_map(sub, xs[idx], w0, lam=0.0)
        wh[g] = wg
        trinv[g] = np.trace(np.linalg.inv(Cg + 1e-6 * np.eye(K)))
        del sub
    used = np.flatnonzero(n > 0)
    if len(used) > 1:
        tau2 = float(np.maximum(0.0, np.mean([((wh[g] - w0) ** 2).sum() - trinv[g] for g in used]) / K))
    else:
        tau2 = 0.0
    lam = np.inf if tau2 <= 0 else 1.0 / tau2
    W = np.tile(w0, (G, 1))
    if np.isfinite(lam):
        for g in used:
            idx = np.flatnonzero(gate == g)
            W[g], _ = poe_map(LP[:, idx, :], xs[idx], w0, lam=lam)
    info = dict(w0=w0.tolist(), tau2=tau2, lam=None if not np.isfinite(lam) else lam,
                n=n.tolist(), dist=[float(np.sqrt(((W[g] - w0) ** 2).sum())) for g in range(G)])
    if verbose:
        print(f"  gate G={G} tau2={tau2:.5f} used={len(used)} n={n.tolist()}", flush=True)
    return W, info


def mix_logp(LP, gate, W):
    """log-prob normalizada por fila: L[i] = sum_k W[gate_i,k] LP_k[i]. LP [K,m,V] -> [m,V] float64."""
    K, m, V = LP.shape
    L = np.zeros((m, V), dtype=np.float64)
    for k in range(K):
        L += W[gate, k][:, None] * LP[k]
    L -= L.max(1, keepdims=True)
    L -= np.log(np.exp(L).sum(1, keepdims=True))
    return L


# ------------------------------------------------------------ detector causal de dominio
@nb.njit(cache=True)
def window_feats(ids, pos, cls, W, nl, sp, out):
    """out[i] = [fraccion de cada clase 1..8 en ids[p-W:p], fraccion de espacios de sangria]. Solo historial (causal)."""
    for i in range(pos.shape[0]):
        p = pos[i]
        lo = p - W if p - W > 0 else 0
        cnt = np.zeros(8)
        lead = False
        ind = 0
        n = p - lo
        for t in range(lo, p):
            c = ids[t]
            k = cls[c]
            if k >= 1:
                cnt[k - 1] += 1.0
            if c == nl:
                lead = True
            elif lead:
                if c == sp:
                    ind += 1
                else:
                    lead = False
        if n > 0:
            for j in range(8):
                out[i, j] = cnt[j] / n
            out[i, 8] = ind / n
        else:
            for j in range(9):
                out[i, j] = 0.0


def gate_features(ids, pos, cls, nl, sp, W=500):
    pos = np.asarray(pos, dtype=np.int64)
    out = np.zeros((len(pos), 9))
    window_feats(np.asarray(ids), pos, cls, W, nl, sp, out)
    return out


def kmeans(X, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        D = ((X[:, None, :] - C[None]) ** 2).sum(2)
        lab = D.argmin(1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C, atol=1e-10):
            break
        C = newC
    D = ((X[:, None, :] - C[None]) ** 2).sum(2)
    return C, D.argmin(1), float(D.min(1).sum())


class CausalDomainGate:
    """Compuerta causal: centroides k-means sobre rasgos estandarizados; k in 1..kmax elegido por BIC."""

    def __init__(self, cls, nl, sp, W=500, kmax=8):
        self.cls, self.nl, self.sp, self.W, self.kmax = cls, nl, sp, W, kmax
        self.C = None
        self.mu = None
        self.sd = None

    def fit(self, train, cut, n_samp=60_000, seed=0):
        rng = np.random.default_rng(seed)
        pos = np.sort(rng.choice(np.arange(self.W, cut), size=min(n_samp, cut - self.W), replace=False))
        F = gate_features(train, pos, self.cls, self.nl, self.sp, self.W)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-9
        X = (F - self.mu) / self.sd
        n, d = X.shape
        best = None; bics = {}
        for k in range(1, self.kmax + 1):
            C, lab, sse = kmeans(X, k, seed=seed)
            bic = n * d * np.log(max(sse, 1e-12) / (n * d)) + k * d * np.log(n)
            bics[k] = round(float(bic), 1)
            if best is None or bic < best[0]:
                best = (bic, k, C)
        self.k = best[1]; self.C = best[2]; self.bics = bics
        return self

    @property
    def G(self):
        return self.k

    def label(self, ids, pos):
        F = gate_features(ids, pos, self.cls, self.nl, self.sp, self.W)
        X = (F - self.mu) / self.sd
        D = ((X[:, None, :] - self.C[None]) ** 2).sum(2)
        return D.argmin(1).astype(np.int64)

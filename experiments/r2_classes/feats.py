"""R2 'classes': agrupacion de palabras y de caracteres POR CONTEO, sin gradientes.

Palabras (idea 1):
  - Palabras = rachas de letras (mismo criterio que core._atoms). Se toman las N = 2000 mas frecuentes del
    tramo de conteo (N es constante libre, ver informe).
  - Matriz de co-ocurrencia con vecino izquierdo y derecho (N x 2N), PPMI, SVD truncada a d=32 (d: constante libre),
    filas normalizadas (coseno).
  - k-means (Lloyd, init k-means++ con semilla 0, max 100 iteraciones o punto fijo) con k = round(sqrt(N)) (regla de datos).
  - Palabras raras (fuera del top-N): cluster por forma = k + ((penultima*31 + ultima) mod 32). Sin parametros.
  - Atomos: pc1 = cluster de la palabra previa completa, pc2 = cluster de 2 palabras atras (sentinela k+32 al inicio).

Caracteres (idea 2):
  - Mismo metodo sobre la matriz de bigramas de caracteres (V x 2V, vecinos izq/der), SVD d=16, k-means.
  - k: 'auto' = el de menor BIC sobre la rejilla {4,8,16,32,64} de un modelo de clases de bigramas
    (log-verosimilitud de bigramas con transicion de clase + emision de caracter; penalizacion 0.5*(k^2+V)*log2 N).
    O bien k fijo (ablacion).
  - La tabla resultante (cls[c] = clase+1; 0 = pad) reemplaza la tabla de 8 clases a mano de core.build_char_tables.
"""
import numpy as np
import numba as nb


# ---------------------------------------------------------------- palabras (numba)
@nb.njit(cache=True)
def _word_stream(ids, isletter):
    """Hash de cada palabra completa (misma formula que core._atoms) y codigo de sus 2 ultimos chars."""
    n = ids.shape[0]
    wh = np.zeros(n, dtype=np.uint64)
    wc = np.zeros(n, dtype=np.int64)
    m = 0
    wp = np.uint64(0)
    a1 = np.int64(255)
    a2 = np.int64(255)
    wlen = 0
    P = np.uint64(1099511628211)
    for t in range(n):
        c = ids[t]
        if isletter[c] == 1:
            wp = (wp + np.uint64(c) + np.uint64(1)) * P
            a2 = a1
            a1 = np.int64(c)
            wlen += 1
        else:
            if wlen > 0:
                wh[m] = wp
                wc[m] = a2 * 256 + a1
                m += 1
            wp = np.uint64(0)
            a1 = np.int64(255)
            a2 = np.int64(255)
            wlen = 0
    return wh[:m], wc[:m]


@nb.njit(cache=True)
def _lookup(keys, h):
    lo = 0
    hi = keys.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < h:
            lo = mid + 1
        else:
            hi = mid
    if lo < keys.shape[0] and keys[lo] == h:
        return lo
    return -1


@nb.njit(cache=True)
def _pcs(ids, isletter, keys, vals, rare_base, none_id, out):
    """out[t] = (cluster palabra previa completa, cluster de 2 atras) usando solo ids[:t]."""
    n = ids.shape[0]
    wp = np.uint64(0)
    a1 = np.int64(255)
    a2 = np.int64(255)
    wlen = 0
    pc1 = none_id
    pc2 = none_id
    P = np.uint64(1099511628211)
    for t in range(n):
        out[t, 0] = pc1
        out[t, 1] = pc2
        c = ids[t]
        if isletter[c] == 1:
            wp = (wp + np.uint64(c) + np.uint64(1)) * P
            a2 = a1
            a1 = np.int64(c)
            wlen += 1
        else:
            if wlen > 0:
                j = _lookup(keys, wp)
                if j >= 0:
                    cl = vals[j]
                else:
                    cl = rare_base + ((a2 * 31 + a1) % 32)
                pc2 = pc1
                pc1 = cl
            wp = np.uint64(0)
            a1 = np.int64(255)
            a2 = np.int64(255)
            wlen = 0


# ---------------------------------------------------------------- comun (numpy)
def _ppmi(X):
    tot = X.sum()
    r = X.sum(1)
    c = X.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(X * tot / np.outer(r, c))
    P = np.where(X > 0, np.maximum(np.nan_to_num(pmi, neginf=0.0, posinf=0.0), 0.0), 0.0)
    return P


def _embed(X, d):
    P = _ppmi(X)
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    d = min(d, len(S))
    E = U[:, :d] * S[:d]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E


def _kmeans(X, k, seed=0, max_iter=100):
    """Lloyd con init k-means++ (semilla fija). Para cuando las asignaciones no cambian (punto fijo)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    C = [X[rng.integers(n)]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        s = d2.sum()
        j = rng.choice(n, p=d2 / s) if s > 0 else rng.integers(n)
        C.append(X[j])
        d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
    C = np.array(C)
    lab = None
    it = 0
    for it in range(1, max_iter + 1):
        D = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        new = D.argmin(1)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab, it


# ---------------------------------------------------------------- idea 1: palabras
class WordClusters:
    def __init__(self, isletter, N=2000, d=32):
        self.isletter, self.N, self.d = isletter, N, d

    def fit(self, train):
        wh, wc = _word_stream(train, self.isletter)
        uniq, inv, cnts = np.unique(wh, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        N = min(self.N, len(uniq))
        order = np.argsort(-cnts, kind="stable")[:N]
        rank = np.full(len(uniq), -1, np.int64)
        rank[order] = np.arange(N)
        s = rank[inv]
        # co-ocurrencia con vecino izquierdo (cols 0..N-1) y derecho (cols N..2N-1)
        l_i, l_a = s[1:], s[:-1]
        mL = (l_i >= 0) & (l_a >= 0)
        r_i, r_b = s[:-1], s[1:]
        mR = (r_i >= 0) & (r_b >= 0)
        idx = np.concatenate([l_i[mL] * (2 * N) + l_a[mL], r_i[mR] * (2 * N) + N + r_b[mR]])
        X = np.bincount(idx, minlength=N * 2 * N).reshape(N, 2 * N).astype(np.float64)
        E = _embed(X, self.d)
        k = int(round(np.sqrt(N)))
        lab, iters = _kmeans(E, k, seed=0, max_iter=100)
        top = uniq[order]
        o = np.argsort(top)
        self.keys = top[o].astype(np.uint64)
        self.vals = lab[o].astype(np.int64)
        self.k = k
        self.rare_base = k
        self.none_id = k + 32
        self.info = dict(word_N=int(N), word_k=k, word_d=int(E.shape[1]), word_kmeans_iters=int(iters),
                         word_top_coverage=float(cnts[order].sum() / max(1, len(s))),
                         word_types_total=int(len(uniq)), word_tokens=int(len(s)))
        return self

    def pcs(self, ids, out):
        _pcs(ids, self.isletter, self.keys, self.vals, self.rare_base, self.none_id, out)
        return out


# ---------------------------------------------------------------- idea 2: clases de caracter
def _bic(C, u, lab, k, n_tokens):
    V = C.shape[0]
    O = np.zeros((V, k))
    O[np.arange(V), lab] = 1.0
    M = O.T @ C @ O                       # conteos de transicion entre clases
    Ra = M.sum(1)
    colsum = C.sum(0)                     # tokens de cada caracter como segundo elemento
    cu = O.T @ colsum
    with np.errstate(divide="ignore", invalid="ignore"):
        LLt = np.nansum(np.where(M > 0, M * np.log2(M / Ra[:, None]), 0.0))
        LLe = np.nansum(np.where(colsum > 0, colsum * np.log2(colsum / cu[lab]), 0.0))
    LL = LLt + LLe
    params = k * k + V
    return -LL + 0.5 * params * np.log2(n_tokens), LL


def char_classes(train, V, k_choice="auto", d=16, grid=(4, 8, 16, 32, 64), seed=0):
    tr = train.astype(np.int64)
    C = np.bincount(tr[:-1] * V + tr[1:], minlength=V * V).reshape(V, V).astype(np.float64)
    u = np.bincount(tr, minlength=V).astype(np.float64)
    X = np.concatenate([C, C.T], axis=1)  # fila x: [que sigue a x | que precede a x]
    E = _embed(X, d)
    labs, bics = {}, {}
    for k in grid:
        lab, it = _kmeans(E, k, seed=seed, max_iter=100)
        labs[k] = (lab, it)
        bics[k] = float(_bic(C, u, lab, k, len(train))[0])
    if k_choice == "auto":
        k = min(bics, key=bics.get)
    else:
        k = int(k_choice)
        if k not in labs:
            lab, it = _kmeans(E, k, seed=seed, max_iter=100)
            labs[k] = (lab, it)
    lab, it = labs[k]
    cls = np.zeros(256, dtype=np.uint8)
    cls[:V] = (lab + 1).astype(np.uint8)  # 0 reservado para pad
    info = dict(char_k=int(k), char_bic={int(a): round(b, 1) for a, b in bics.items()},
                char_kmeans_iters=int(it))
    return cls, info


# ---------------------------------------------------------------- proveedor para core.Ixchel
class R2Prov:
    def __init__(self, default_cls, isletter, use_words=True, N=2000, d=32, char_k=None, char_d=16):
        self.default_cls, self.isletter = default_cls, isletter
        self.use_words, self.N, self.d = use_words, N, d
        self.char_k, self.char_d = char_k, char_d
        self.cls = default_cls
        self.info = {}

    def fit(self, train, V):
        info = {}
        if self.use_words:
            self.wc = WordClusters(self.isletter, self.N, self.d).fit(train)
            info.update(self.wc.info)
        if self.char_k is None:
            self.cls = self.default_cls
        else:
            self.cls, ci = char_classes(train, V, self.char_k, d=self.char_d)
            info.update(ci)
        self.info = info

    def pcs(self, ids):
        out = np.zeros((len(ids), 2), dtype=np.int64)
        if self.use_words:
            self.wc.pcs(ids, out)
        return out

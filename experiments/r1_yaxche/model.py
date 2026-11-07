"""Yax Che' (MVP): bosque de arboles de contexto aprendidos por MDL, sin gradientes.

Cada arbol aprende la estructura del contexto: nodos internos = preguntas
  (a) "el caracter en t-j pertenece al conjunto S"  (j=1..L; S = un caracter, o una clase)
  (b) "distancia al ultimo espacio/salto de linea" en {d<=k} o {d==k}
Hojas = distribucion de conteo del siguiente caracter, suavizada Witten-Bell jerarquica
(p_hoja = (n + T*p_padre)/(N+T), T = numero de tipos observados en la hoja; sin parametros).
Division codiciosa por ganancia en codelength KT (alpha=1/2) menos el costo de describir la
pregunta, ln(Q). Para la.parada: se divide solo si reduce el codigo total (MDL).
El bosque promedia (pesos iguales) los B arboles, cada uno entrenado en un bootstrap de
posiciones y con un subconjunto aleatorio de features.
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import string, time
import numpy as np
from numba import njit
from scipy.special import gammaln

DMAX = 31  # distancia maxima al espacio considerada (0 = no hay espacio en 31 chars)


def build_classes(itos):
    """Clases de caracteres (solapables permitidas). Devuelve nombres, memb (V,K) uint8."""
    V = len(itos)
    vow = set("aeiouAEIOU")
    defs = {
        "vowel": vow,
        "consonant": set(string.ascii_letters) - vow,
        "upper": set(string.ascii_uppercase),
        "lower": set(string.ascii_lowercase),
        "space_tab": {" ", "\t"},
        "newline": {"\n"},
        "digit": set(string.digits),
        "sent_end": set(".!?"),
        "punct": set(string.punctuation) - set(".!?"),
    }
    chars = set(itos)
    defs["other"] = chars - set().union(*defs.values())
    names, cols = [], []
    for k, s in defs.items():
        col = np.array([c in s for c in itos], dtype=np.uint8)
        if col.sum() > 0:
            names.append(k)
            cols.append(col)
    memb = np.stack(cols, axis=1)
    return names, memb


@njit(cache=True)
def _build_F(x, spm, L):
    """F[pos, j-1] = x[pos-j] (j=1..L), F[pos, L] = distancia al espacio (0 si no hay en DMAX)."""
    n = len(x)
    F = np.zeros((n, L + 1), np.uint8)
    for pos in range(n):
        for j in range(1, L + 1):
            if pos - j >= 0:
                F[pos, j - 1] = x[pos - j]
        for k in range(1, DMAX + 1):
            p = pos - k
            if p < 0:
                break
            if spm[x[p]]:
                F[pos, L] = k
                break
    return F


@njit(cache=True)
def _split_cost(left, nl, cnt, ynz, nyz, N, G, LG, G0, LG0):
    """KT codelength (nats) de los dos hijos de un split; left = conteos del hijo izquierdo."""
    sl = 0.0
    sr = 0.0
    for t in range(nyz):
        y = ynz[t]
        a = left[y]
        b = cnt[y] - a
        sl += G[a] - G0
        sr += G[b] - G0
    Kl = LG[nl] - LG0 - sl
    Kr = LG[N - nl] - LG0 - sr
    return Kl + Kr


@njit(cache=True)
def _build_tree(Fs, ys, V, K, L, memb, cls_list, ncls, feat_allowed, G, LG, G0, LG0, qcost, maxnodes, maxdepth,
                nd_feat, nd_mask, nd_left, nd_right, nd_prob, nd_depth):
    S = Fs.shape[0]
    NF = L + 1
    R = max(V + K, DMAX + 1)
    H = np.zeros((NF, R, V), np.int32)
    rowc = np.zeros((NF, R), np.int64)
    cnt = np.zeros(V, np.int64)
    ynz = np.zeros(V, np.int64)
    idx = np.arange(S)
    st_id = np.zeros(maxnodes, np.int64)
    st_lo = np.zeros(maxnodes, np.int64)
    st_hi = np.zeros(maxnodes, np.int64)
    st_par = np.zeros(maxnodes, np.int64)
    sp = 0
    nn = 1
    nd_left[0] = -1
    nd_right[0] = -1
    nd_depth[0] = 0
    st_id[0] = 0; st_lo[0] = 0; st_hi[0] = S; st_par[0] = -1
    sp = 1
    mask = np.zeros(V, np.uint8)
    unif = np.full(V, 1.0 / V)
    nsplit = 0
    while sp > 0:
        sp -= 1
        nid = st_id[sp]; lo = st_lo[sp]; hi = st_hi[sp]; par = st_par[sp]
        # --- histogramas del nodo
        H[:] = 0
        rowc[:] = 0
        cnt[:] = 0
        for i in range(lo, hi):
            s = idx[i]
            y = ys[s]
            cnt[y] += 1
            for f in range(NF):
                v = Fs[s, f]
                H[f, v, y] += 1
                rowc[f, v] += 1
                if f < L:
                    # clases que contienen a v
                    for q in range(ncls[v]):
                        c = cls_list[v, q]
                        H[f, V + c, y] += 1
                        rowc[f, V + c] += 1
        N = hi - lo
        nyz = 0
        for y in range(V):
            if cnt[y] > 0:
                ynz[nyz] = y
                nyz += 1
        T = nyz
        # --- suavizado Witten-Bell jerarquico (prior = padre)
        if par < 0:
            for y in range(V):
                nd_prob[nid, y] = (cnt[y] + T * unif[y]) / (N + T)
        else:
            for y in range(V):
                nd_prob[nid, y] = (cnt[y] + T * nd_prob[par, y]) / (N + T)
        depth = nd_depth[nid]
        if N < 2 or depth >= maxdepth:
            continue
        # --- mejor split por MDL (KT)
        Kpar = LG[N] - LG0
        for t in range(nyz):
            Kpar -= G[cnt[ynz[t]]] - G0
        best = 0.0
        bf = -1; br = -1; bkind = 0
        for f in range(NF):
            if not feat_allowed[f]:
                continue
            if f < L:
                for r in range(V + K):
                    nl = rowc[f, r]
                    if nl == 0 or nl == N:
                        continue
                    kc = _split_cost(H[f, r], nl, cnt, ynz, nyz, N, G, LG, G0, LG0)
                    gain = Kpar - kc - qcost
                    if gain > best:
                        best = gain; bf = f; br = r; bkind = 0
            else:
                # singletons de distancia
                for d in range(DMAX + 1):
                    nl = rowc[f, d]
                    if nl == 0 or nl == N:
                        continue
                    kc = _split_cost(H[f, d], nl, cnt, ynz, nyz, N, G, LG, G0, LG0)
                    gain = Kpar - kc - qcost
                    if gain > best:
                        best = gain; bf = f; br = d; bkind = 0
                # prefijos {d <= k}
                cum = np.zeros(V, np.int32)
                cn = 0
                for k in range(DMAX):
                    for y in range(V):
                        cum[y] += H[f, k, y]
                    cn += rowc[f, k]
                    if cn == 0 or cn == N:
                        continue
                    kc = _split_cost(cum, cn, cnt, ynz, nyz, N, G, LG, G0, LG0)
                    gain = Kpar - kc - qcost
                    if gain > best:
                        best = gain; bf = f; br = k; bkind = 1
        if bf < 0:
            continue
        # --- mascara del split
        for v in range(V):
            mask[v] = 0
        if bkind == 1:
            for v in range(br + 1):
                mask[v] = 1
        elif bf >= L:
            mask[br] = 1
        elif br < V:
            mask[br] = 1
        else:
            c = br - V
            for v in range(V):
                if memb[v, c]:
                    mask[v] = 1
        nd_feat[nid] = bf
        for v in range(V):
            nd_mask[nid, v] = mask[v]
        # --- particion in situ: izquierda = mask[valor]==1
        i = lo; j = hi - 1
        while i <= j:
            s = idx[i]
            if mask[Fs[s, bf]]:
                i += 1
            else:
                idx[i] = idx[j]; idx[j] = s
                j -= 1
        mid = i
        if nn + 2 > maxnodes:
            raise RuntimeError("maxnodes excedido")
        lc = nn; rc = nn + 1
        nn += 2
        nd_left[nid] = lc; nd_right[nid] = rc
        nd_depth[lc] = depth + 1; nd_depth[rc] = depth + 1
        nd_left[lc] = -1; nd_right[lc] = -1; nd_left[rc] = -1; nd_right[rc] = -1
        st_id[sp] = lc; st_lo[sp] = lo; st_hi[sp] = mid; st_par[sp] = nid; sp += 1
        st_id[sp] = rc; st_lo[sp] = mid; st_hi[sp] = hi; st_par[sp] = nid; sp += 1
        nsplit += 1
    return nn


@njit(cache=True)
def _feat_at(ids, p, L, spm, fv):
    for j in range(1, L + 1):
        fv[j - 1] = ids[p - j] if p - j >= 0 else 0
    d = 0
    for k in range(1, DMAX + 1):
        q = p - k
        if q < 0:
            break
        if spm[ids[q]]:
            d = k
            break
    fv[L] = d


@njit(cache=True)
def _predict(ids, pos, roots, feat, mask, left, right, prob, depth, maxdep, L, spm, V, out):
    m = len(pos)
    fv = np.zeros(L + 1, np.int64)
    nb = len(roots)
    for i in range(m):
        _feat_at(ids, pos[i], L, spm, fv)
        acc = np.zeros(V, np.float64)
        for b in range(nb):
            nd = roots[b]
            while left[nd] >= 0 and depth[nd] < maxdep:
                if mask[nd, fv[feat[nd]]]:
                    nd = left[nd]
                else:
                    nd = right[nd]
            for y in range(V):
                acc[y] += prob[nd, y]
        for y in range(V):
            out[i, y] = acc[y] / nb


class Yaxche:
    def __init__(self, itos, B=16, S=1_000_000, L=16, feat_frac=1.0, max_depth=64, seed=0,
                 maxnodes=300_000, name=None):
        self.itos = itos
        self.B = B
        self.S = S
        self.L = L
        self.feat_frac = feat_frac
        self.max_depth = max_depth
        self.seed = seed
        self.maxnodes = maxnodes
        self.name = name or f"Yaxche-MDL-B{B}"
        self.B_use = B
        self.tree_info = []

    def fit(self, train, V):
        self.V = V
        self.spm = np.array([c in (" ", "\n") for c in self.itos], dtype=np.uint8)
        self.cls_names, memb = build_classes(self.itos)
        self.memb = memb
        self.K = memb.shape[1]
        # lista de clases por caracter (para histogramas rapidos)
        self.ncls = memb.sum(1).astype(np.int64)
        self.cls_list = np.zeros((V, max(1, int(self.ncls.max()))), np.int64)
        for v in range(V):
            cs = np.nonzero(memb[v])[0]
            self.cls_list[v, :len(cs)] = cs
        rng = np.random.default_rng(self.seed)
        x = np.ascontiguousarray(train)
        n = len(x)
        t0 = time.time()
        F = _build_F(x, self.spm, self.L)
        self.F_build_s = time.time() - t0
        NF = self.L + 1
        Q = (V + self.K) * self.L + (DMAX + 1) + DMAX  # candidatos totales
        self.Q = Q
        qcost = float(np.log(Q))
        G = gammaln(np.arange(self.S + 1) + 0.5)
        LG = gammaln(np.arange(self.S + 1) + V / 2.0)
        G0 = float(gammaln(0.5)); LG0 = float(gammaln(V / 2.0))
        nfeat_sub = max(1, int(round(self.feat_frac * NF)))
        self.trees = []
        self.tree_info = []
        # buffers reutilizables (cada entrada se escribe antes de leerse)
        nd_feat = np.zeros(self.maxnodes, np.int64)
        nd_mask = np.zeros((self.maxnodes, V), np.uint8)
        nd_left = np.zeros(self.maxnodes, np.int64)
        nd_right = np.zeros(self.maxnodes, np.int64)
        nd_prob = np.zeros((self.maxnodes, V), np.float32)
        nd_depth = np.zeros(self.maxnodes, np.int64)
        for b in range(self.B):
            t1 = time.time()
            pos_s = rng.integers(self.L, n, size=self.S)  # bootstrap de posiciones
            Fs = F[pos_s]
            ys = x[pos_s].astype(np.int64)
            allowed = np.zeros(NF, dtype=np.bool_)
            allowed[rng.choice(NF, size=nfeat_sub, replace=False)] = True
            nn = _build_tree(Fs, ys, V, self.K, self.L, memb, self.cls_list, self.ncls, allowed, G, LG, G0, LG0, qcost,
                             self.maxnodes, self.max_depth,
                             nd_feat, nd_mask, nd_left, nd_right, nd_prob, nd_depth)
            leaves = int(((nd_left[:nn]) < 0).sum())
            dep = nd_depth[:nn][nd_left[:nn] < 0]
            self.trees.append(dict(feat=nd_feat[:nn].copy(), mask=nd_mask[:nn].copy(),
                                   left=nd_left[:nn].copy(), right=nd_right[:nn].copy(),
                                   prob=nd_prob[:nn].copy(), depth=nd_depth[:nn].copy()))
            self.tree_info.append(dict(nodes=int(nn), leaves=leaves, max_leaf_depth=int(dep.max()),
                                       mean_leaf_depth=float(dep.mean()),
                                       build_s=time.time() - t1,
                                       feats_used=int(allowed.sum())))
        self._concat()
        self.fit_s_per_tree = [t["build_s"] for t in self.tree_info]
        self.fit_wall_s = float(self.F_build_s + sum(self.fit_s_per_tree))

    def _concat(self):
        feats, masks, lefts, rights, probs, depths, roots = [], [], [], [], [], [], []
        off = 0
        for t in self.trees:
            nn = len(t["feat"])
            roots.append(off)
            lf = t["left"].copy(); rt = t["right"].copy()
            lf[lf >= 0] += off; rt[rt >= 0] += off
            feats.append(t["feat"]); masks.append(t["mask"]); lefts.append(lf); rights.append(rt)
            probs.append(t["prob"]); depths.append(t["depth"])
            off += nn
        self.g = dict(feat=np.concatenate(feats).astype(np.int16), mask=np.concatenate(masks),
                      left=np.concatenate(lefts), right=np.concatenate(rights),
                      prob=np.concatenate(probs), depth=np.concatenate(depths).astype(np.int16))
        self.roots_all = np.array(roots, dtype=np.int64)

    def set_B(self, b):
        self.B_use = b

    def n_params(self):
        leaves = sum(self.tree_info[b]["leaves"] for b in range(self.B_use))
        return int(leaves * self.V)

    def predict(self, ids, pos, max_depth=None):
        pos = np.asarray(pos, dtype=np.int64)
        out = np.zeros((len(pos), self.V), np.float64)
        md = self.max_depth + 1 if max_depth is None else max_depth
        g = self.g
        _predict(np.ascontiguousarray(ids), pos, self.roots_all[:self.B_use], g["feat"], g["mask"],
                 g["left"], g["right"], g["prob"], g["depth"], int(md), self.L, self.spm, self.V, out)
        return out.astype(np.float32)

import os, sys, json, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import gatepoe as GP
from mixing import poe_global
rng = np.random.default_rng(0)
K, m, V = 5, 4000, 20
LP = np.log(rng.dirichlet(np.ones(V), size=(K, m)) + 1e-12).astype(np.float32)
xs = rng.integers(0, V, m)
w_ref = poe_global(LP, xs, iters=60)
w_new, C = GP.poe_map(LP, xs, np.zeros(K), lam=0.0)
print("poe_global", np.round(w_ref, 5)); print("poe_map   ", np.round(w_new, 5))
# gradient check at MLE
S = np.stack([LP[k, np.arange(m), xs] for k in range(K)], 1).sum(0)
F, g, C = GP._stats(LP, xs, w_new, S); print("grad at MLE (should be ~0):", np.abs(g).max())
# finite-difference check of C = -Hessian
h = 1e-4; Cfd = np.zeros((K, K))
for j in range(K):
    e = np.zeros(K); e[j] = h
    _, gp_, _ = GP._stats(LP, xs, w_new + e, S); _, gm_, _ = GP._stats(LP, xs, w_new - e, S)
    Cfd[:, j] = -(gp_ - gm_) / (2 * h)  # d(grad)/dw = -C
print("max |C - fd(-Hess)|:", np.abs(C - Cfd).max(), " scale", np.abs(C).max())
# gates + shrinkage
gate = rng.integers(0, 3, m); W, info = GP.fit_gated(LP, xs, gate, 3, verbose=True)
print("tau2", info["tau2"], "dist", np.round(info["dist"], 4))
# shrinkage with a fixed-small random gate: W should not diverge
gate2 = rng.integers(0, 40, m); W2, info2 = GP.fit_gated(LP, xs, gate2, 40); print("G=40 tau2", info2["tau2"], "max|w|", np.abs(W2).max())
# causal detector on real data slice
from core_v04 import build_char_tables
itos = json.load(open(os.path.join(HERE, "..", "..", "data", "mix15", "vocab.json")))
tr = np.load(os.path.join(HERE, "..", "..", "data", "mix15", "train.npy"))
cls, isl, isnl = build_char_tables(itos)
t = time.time(); gt = GP.CausalDomainGate(cls, itos.index("\n"), itos.index(" "), W=500).fit(tr, 3_000_000, n_samp=20000)
print("detector k", gt.k, "bics", gt.bics, "time", round(time.time() - t, 2))
lab = gt.label(tr[3_000_000:], np.arange(500, 4000))
print("label counts", np.bincount(lab, minlength=gt.k).tolist())
# causality: label at p must not depend on ids[p:]
ids2 = tr[3_000_000:].copy(); p = 2000; ids2[p:] = 7
print("causal ok:", np.array_equal(gt.label(tr[3_000_000:], [p]), gt.label(ids2, [p])))

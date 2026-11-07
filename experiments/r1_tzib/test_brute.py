import os, sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, harness
from model import Tzib, DMAX
train, val, itos = harness.load(); V = len(itos); T = train.astype(np.int64); n = len(T)
mk = Tzib("KN-5"); mk.fit(train, V); me = Tzib("EXCL-5"); me.fit(train, V)
Dr, Dc = mk.meta["Dr"], mk.meta["Dc"]
def occ(ctx):
    L = len(ctx); m = np.ones(n - L + 1, bool)
    for k, c in enumerate(ctx): m &= T[k:n - L + 1 + k] == c
    j = np.nonzero(m)[0]
    return j[j + L < n]           # occurrences with a next char
def brute(ctxseq):
    """ctxseq: context array (last chars). returns (raw counts per level, cont counts per level, Ls)."""
    raws, conts = [np.bincount(T, minlength=V).astype(float)], []
    cont0 = np.zeros(V); 
    # cont0: distinct w before s (bigrams)
    bg = np.unique(T[:-1] * V + T[1:]); cont0 = np.bincount(bg % V, minlength=V).astype(float)
    conts.append(cont0)
    Ls = 0
    for L in range(1, min(len(ctxseq), 40) + 1):
        ctx = ctxseq[-L:]
        j = occ(ctx)
        if len(j) == 0: break
        r = np.bincount(T[j + L], minlength=V).astype(float)
        jj = j[j >= 1]; s = T[jj + L]; w = T[jj - 1]
        pairs = np.unique(np.stack([w, s], 1), axis=0) if len(jj) else np.zeros((0, 2), np.int64)
        c = np.bincount(pairs[:, 1], minlength=V).astype(float) if len(pairs) else np.zeros(V)
        raws.append(r); conts.append(c); Ls = L
    return raws, conts, Ls
def kn5(ctxseq):
    raws, conts, Ls = brute(ctxseq)
    t = min(Ls, 5)
    A0 = conts[0].sum(); U0c = (conts[0] > 0).sum()
    Q = np.maximum(conts[0] - Dc[0], 0) / A0 + Dc[0] * U0c / A0 / V
    for L in range(1, t):
        c = conts[L]; A = c.sum(); Uc = (c > 0).sum()
        Q = np.maximum(c - Dc[L], 0) / A + Dc[L] * Uc / A * Q
    r = raws[t]; N = r.sum(); U = (r > 0).sum()
    return np.maximum(r - Dr[t], 0) / N + Dr[t] * U / N * Q, Ls
def excl5(ctxseq):
    raws, conts, Ls = brute(ctxseq); t = min(Ls, 5)
    E = np.zeros(V, bool); p = np.zeros(V); R = 1.0
    for L in range(t, -1, -1):
        r = raws[L].copy(); r[E] = 0
        if L == 0:
            nonex = (~E).sum(); u = 1.0 / nonex; N = r.sum(); U = (r > 0).sum(); den = N + U
            p[~E] += R * (r[~E] + U * u) / den
        else:
            N = r.sum(); U = (r > 0).sum(); den = N + U
            if den > 0:
                p[r > 0] += R * r[r > 0] / den; R *= U / den
            E |= raws[L] > 0
    return p, Ls
rng = np.random.default_rng(3)
pos = rng.integers(200, len(val) - 1, 6)
PK = mk.predict(val, pos); PE = me.predict(val, pos)
for i, p in enumerate(pos):
    ctx = val[:p].astype(np.int64)[-60:]
    bk, Ls = kn5(ctx); be, _ = excl5(ctx)
    print(f"pos {p} Ls={Ls} KN maxdiff={np.abs(bk-PK[i]).max():.2e}  EXCL maxdiff={np.abs(be-PE[i]).max():.2e}  KNsum={bk.sum():.6f}", flush=True)

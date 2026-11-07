"""Analisis del seguimiento (a): AUC por degradacion para: Ixchel con cache, Ixchel sin cache, KN (A), detector de repeticion (gap) y combinado (rango-maximo).
Uso: python -I niche1_nocache_analysis.py"""
import os, json
import numpy as np
from scipy.stats import rankdata, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
old = np.load(os.path.join(R, "niche1_scores.npz")); nc = np.load(os.path.join(R, "niche1_nocache_scores.npz"))
rng = np.random.default_rng(0); B = 2000; n = len(old["clean"])
def feats(k):  # columnas: con cache, sin cache, KN A, gap (sin cache - con cache)
    return np.stack([old[k][:, 0], nc[k][:, 0], nc[k][:, 1], nc[k][:, 0] - old[k][:, 0]], 1)
F = {k: feats(k) for k in ("clean", "noise2", "noise10", "wshuf", "repeat", "splice")}
names = ["Ixchel con cache", "Ixchel sin cache", "KN (A)", "gap (repeticion)", "combinado max-rango"]
def combined(clean, deg):
    # rangos calculados sobre la union limpio+degradado (sin constantes): alto en bpc sin cache O alto en gap
    a = np.concatenate([clean, deg]); r1 = rankdata(a[:, 1]) / len(a); r2 = rankdata(a[:, 3]) / len(a); c = np.maximum(r1, r2); return c[:len(clean)], c[len(clean):]
def auc(neg, pos):
    r = rankdata(np.concatenate([neg, pos])); n1, n2 = len(neg), len(pos); return (r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)
lines = ["## Seguimiento (a): AUC limpio vs degradado con IC95% bootstrap por documento", "| degradacion | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
out = {}
for k in ("noise2", "noise10", "wshuf", "repeat", "splice"):
    row = []; out[k] = {}
    for c in range(5):
        def get(ix):
            cl, dg = F["clean"][ix], F[k][ix]
            if c < 4: return auc(cl[:, c], dg[:, c])
            a, b = combined(cl, dg); return auc(a, b)
        a = get(np.arange(n)); bs = [get(rng.integers(0, n, n)) for _ in range(B // 4)]; lo, hi = np.percentile(bs, [2.5, 97.5])
        out[k][names[c]] = dict(auc=float(a), lo=float(lo), hi=float(hi)); row.append(f"{a:.3f} [{lo:.3f}, {hi:.3f}]")
    lines.append(f"| {k} | " + " | ".join(row) + " |")
qf = os.path.join(R, "niche1_qwen_bpc.npy")
if os.path.exists(qf):
    q = np.load(qf); m = len(q); lines += ["", f"## E2: Spearman con bpc de Qwen2.5-1.5B (n={m}), IC95% bootstrap", "| puntuador | Spearman | IC95% |", "|---|---|---|"]
    for c in range(3):
        x = F["clean"][:m, c]; rho = spearmanr(x, q).statistic; bs = []
        for _ in range(B):
            ix = rng.integers(0, m, m); bs.append(spearmanr(x[ix], q[ix]).statistic)
        lo, hi = np.percentile(bs, [2.5, 97.5]); lines.append(f"| {names[c]} | {rho:.3f} | [{lo:.3f}, {hi:.3f}] |"); out.setdefault("E2", {})[names[c]] = dict(rho=float(rho), lo=float(lo), hi=float(hi))
lines += ["", "bpc medio limpio: " + ", ".join(f"{names[c]} {F['clean'][:, c].mean():.3f}" for c in range(4))]
txt = "\n".join(lines); print(txt); open(os.path.join(R, "niche1_nocache_analysis.md"), "w", encoding="utf-8").write(txt); json.dump(out, open(os.path.join(R, "niche1_nocache_analysis.json"), "w"), indent=1)

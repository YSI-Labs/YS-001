"""NICHO 1: analisis (AUC + IC bootstrap por documento, precision pareada, Spearman vs Qwen). Uso: python -I niche1_analysis.py"""
import os, json
import numpy as np
from scipy.stats import rankdata, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
S = np.load(os.path.join(R, "niche1_scores.npz")); names = {0: "Ixchel completo", 1: "n-grama KN (A)", 2: "zlib"}
rng = np.random.default_rng(0); B = 2000

def auc(neg, pos):  # P(score_pos > score_neg), pos = degradado
    r = rankdata(np.concatenate([neg, pos])); n1, n2 = len(neg), len(pos)
    return (r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)

out = {"E1": {}, "E2": {}}; clean = S["clean"]; n = len(clean)
lines = ["## E1: AUC (limpio vs degradado; 0.5 = azar) con IC95% bootstrap por documento; entre parentesis, precision pareada (mismo doc)"]
lines.append("| degradacion | " + " | ".join(names.values()) + " |"); lines.append("|---|---|---|---|")
for k in ("noise2", "noise10", "wshuf", "repeat", "splice"):
    deg = S[k]; row = []; out["E1"][k] = {}
    for c in range(3):
        a = auc(clean[:, c], deg[:, c]); pair = float((deg[:, c] > clean[:, c]).mean())
        bs = []
        for _ in range(B):
            ix = rng.integers(0, n, n); bs.append(auc(clean[ix, c], deg[ix, c]))
        lo, hi = np.percentile(bs, [2.5, 97.5]); out["E1"][k][names[c]] = dict(auc=float(a), lo=float(lo), hi=float(hi), paired=pair)
        row.append(f"{a:.3f} [{lo:.3f}, {hi:.3f}] ({pair:.2f})")
    lines.append(f"| {k} | " + " | ".join(row) + " |")
qf = os.path.join(R, "niche1_qwen_bpc.npy")
if os.path.exists(qf):
    q = np.load(qf); m = len(q); lines.append(f"\n## E2: Spearman con el bpc de Qwen2.5-1.5B (n={m} docs C4 limpios), IC95% bootstrap; solape de los 20% peores")
    lines.append("| puntuador | Spearman | IC95% | solape top-20% peor |"); lines.append("|---|---|---|---|")
    k20 = m // 5; worst_q = set(np.argsort(-q)[:k20])
    for c in range(3):
        x = clean[:m, c]; rho = spearmanr(x, q).statistic; bs = []
        for _ in range(B):
            ix = rng.integers(0, m, m); bs.append(spearmanr(x[ix], q[ix]).statistic)
        lo, hi = np.percentile(bs, [2.5, 97.5]); ov = len(worst_q & set(np.argsort(-x)[:k20])) / k20
        out["E2"][names[c]] = dict(rho=float(rho), lo=float(lo), hi=float(hi), overlap20=float(ov)); lines.append(f"| {names[c]} | {rho:.3f} | [{lo:.3f}, {hi:.3f}] | {ov:.2f} (azar 0.20) |")
lines.append("\nbpc medio (limpio): " + ", ".join(f"{names[c]} {clean[:, c].mean():.3f}" for c in range(3)))
txt = "\n".join(lines); print(txt); open(os.path.join(R, "niche1_analysis.md"), "w", encoding="utf-8").write(txt); json.dump(out, open(os.path.join(R, "niche1_analysis.json"), "w"), indent=1)

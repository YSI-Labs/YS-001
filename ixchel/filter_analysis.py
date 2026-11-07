"""Analisis del experimento entrenar con/sin filtro: bpc medio por brazo (2 semillas), dispersion entre semillas y diferencias pareadas por documento contra un brazo de referencia."""
import os, json, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "filter")
arms = ["N_all", "N_rand50", "N_low50", "N_high50", "C_unf", "C_filtA", "C_filtFull", "C_oracle"]
nll = {a: np.stack([np.load(os.path.join(R, f"{a}_s{s}_nll.npy")) for s in (0, 1)]) for a in arms}  # [seed, doc]
lines = ["| brazo | bpc semilla 0 | semilla 1 | media | dispersion entre semillas |", "|---|---|---|---|---|"]
for a in arms:
    m = nll[a].mean(1); lines.append(f"| {a} | {m[0]:.4f} | {m[1]:.4f} | {m.mean():.4f} | {abs(m[0]-m[1]):.4f} |")
def diff(a, b):
    d = nll[a].mean(0) - nll[b].mean(0); se = d.std(ddof=1) / np.sqrt(len(d)); return d.mean(), se
lines += ["", "Diferencias pareadas por documento (media de 2 semillas; negativo = el primero es mejor), con error estandar por documento:", "| comparacion | diferencia | SE |", "|---|---|---|"]
for a, b in [("N_rand50", "N_all"), ("N_low50", "N_rand50"), ("N_high50", "N_rand50"), ("C_unf", "C_oracle"), ("C_filtA", "C_unf"), ("C_filtFull", "C_unf"), ("C_filtA", "C_oracle"), ("C_filtFull", "C_oracle"), ("C_filtA", "C_filtFull")]:
    d, se = diff(a, b); lines.append(f"| {a} - {b} | {d:+.4f} | {se:.4f} |")
txt = "\n".join(lines); print(txt); open(os.path.join(R, "..", "filter_analysis.md"), "w", encoding="utf-8").write(txt)

"""Figuras del preprint. Todos los datos se copian de las tablas del texto (que a su vez salen de ixchel/results/*.json, RESULTS_text8.md, niche4_*.json, filter_analysis.md)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
C = dict(ix="#0072B2", tl="#E69F00", qw="#009E73", ppmd="#CC79A7", xz="#7F7F7F", ref="#444444")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200); plt.close(fig)


# --- Fig 1: text8 (tabla §9.1)
n = np.array([1, 3, 10, 30, 90]); full = [1.766, 1.665, 1.559, 1.486, 1.419]; A = [2.042, 1.899, 1.760, 1.668]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ax.plot(n, full, "o-", color=C["ix"], label="Ixchel (≈ v0.3, conjunto completo)")
ax.plot(n[:4], A, "s--", color=C["xz"], label="solo cadena contigua KN (orden ≤ 8)")
for y, lab, dy in ((1.08, "Transformer-XL ≈ 1.08", 0.012), (1.27, "mLSTM ≈ 1.27", -0.045), (1.29, "HM-LSTM ≈ 1.29", 0.012)):
    ax.axhline(y, color=C["ref"], ls=":", lw=1); ax.text(1.05, y + dy, lab + " (publicado, entrenado con 90 M)", fontsize=7.5, color=C["ref"])
ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([f"{v} M" for v in n]); ax.set_xlabel("caracteres de entrenamiento"); ax.set_ylabel("bpc en el test estándar de text8")
ax.legend(frameon=False, loc="upper right", fontsize=8); ax.set_ylim(1.0, 2.15)
save(fig, "fig1_text8.png")

# --- Fig 2: bpc por dominio (tabla §10.2)
dom = ["web", "código", "matemáticas", "chat", "libros"]
ix = [1.595, 1.309, 1.413, 1.429, 2.053]; tl = [0.905, 0.798, 0.815, 0.757, 1.380]; qw = [0.925, 0.692, 0.659, 0.696, 1.294]
x = np.arange(5); w = 0.27; fig, ax = plt.subplots(figsize=(6.3, 3.3))
for k, (v, lab, col) in enumerate(((ix, "Ixchel v0.5 (1.17 mil millones de celdas)", C["ix"]), (tl, "TinyLlama-1.1B", C["tl"]), (qw, "Qwen2.5-1.5B", C["qw"]))):
    b = ax.bar(x + (k - 1) * w, v, w, color=col, label=lab)
    for r, val in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(dom); ax.set_ylabel("bits por carácter (menos es mejor)"); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_ylim(0, 2.45)
save(fig, "fig2_dominios.png")

# --- Fig 3: compresión (bpc = bytes*8/caracteres; §13.2 y §13.3)
chars = [1007363, 1023551, 730400, 634078, 536038]
ixb = [200012, 167247, 130440, 113173, 139418]; ppmd = [270013, 182803, 161688, 165447, 152278]; xz = [322348, 204620, 194860, 197148, 188264]
f = lambda b: [bb * 8 / c for bb, c in zip(b, chars)]
fig, ax = plt.subplots(figsize=(6.3, 3.3))
for k, (v, lab, col) in enumerate(((f(ixb), "Ixchel (cota de código ideal)", C["ix"]), (f(ppmd), "PPMd (mejor de órdenes 8, 16, 32)", C["ppmd"]), (f(xz), "xz 9e", C["xz"]))):
    b = ax.bar(x + (k - 1) * w, v, w, color=col, label=lab)
    for r, val in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(dom); ax.set_ylabel("bits por carácter comprimido (menos es mejor)"); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_ylim(0, 3.3)
save(fig, "fig3_compresion.png")

# --- Fig 4: filtrado (tabla §13.3; puntos = semillas 0 y 1; rombo = media)
arms = ["limpio" + chr(10) + "todos", "limpio" + chr(10) + "50 % al azar", "limpio" + chr(10) + "50 % más" + chr(10) + "fácil", "limpio" + chr(10) + "50 % más" + chr(10) + "difícil", "30 % basura" + chr(10) + "sin filtrar", "filtro" + chr(10) + "n-grama KN", "filtro" + chr(10) + "Ixchel", "oráculo"]
s0 = [2.2085, 2.2156, 2.2379, 2.2304, 2.2397, 2.2172, 2.2237, 2.2145]; s1 = [2.2175, 2.2180, 2.2365, 2.2295, 2.2436, 2.2279, 2.2336, 2.2226]
mean = [(a + b) / 2 for a, b in zip(s0, s1)]; xs = np.arange(8); cols = [C["xz"]] * 4 + ["#D55E00", C["ix"], C["ppmd"], C["qw"]]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for i in range(8): ax.plot([xs[i], xs[i]], [s0[i], s1[i]], color=cols[i], lw=2, alpha=0.6, zorder=2)
ax.scatter(xs, s0, color="black", s=14, zorder=3, label="semilla 0"); ax.scatter(xs, s1, color="white", edgecolor="black", s=16, zorder=3, label="semilla 1")
ax.scatter(xs, mean, marker="D", s=40, c=cols, edgecolor="black", zorder=4, label="media")
ax.set_ylim(2.205, 2.247); ax.set_xticks(xs); ax.set_xticklabels(arms, fontsize=7); ax.set_ylabel("bpc sobre documentos limpios no vistos" + chr(10) + "(eje truncado; menos es mejor)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.axvline(3.5, color="black", lw=0.6, ls=":")
save(fig, "fig4_filtrado.png")
print("figuras:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))

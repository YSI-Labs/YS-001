"""NICHO 4: PPMd (variante H de 7-Zip, via pyppmd) sobre los MISMOS archivos val_* completos de niche4_compress (competidor de modelado de contexto sin gradientes).
Ordenes 8, 16 y 32 con 256 MB de memoria. Uso: python -I niche4_ppmd.py"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import numpy as np, pyppmd
D = os.path.join(ROOT, "data", "big2"); itos = json.load(open(os.path.join(D, "vocab.json"))); res = {}
for d in ("web", "code", "math", "chat", "books"):
    val = np.load(os.path.join(D, f"val_{d}.npy")); text = "".join(itos[i] for i in val).encode("ascii", "replace"); r = {}
    for o in (8, 16, 32):
        t = time.time(); r[f"ppmd_o{o}"] = len(pyppmd.compress(text, max_order=o, mem_size=256 << 20)); r[f"ppmd_o{o}_s"] = time.time() - t
    r["chars"] = len(text); res[d] = r; print(d, r, flush=True)
json.dump(res, open(os.path.join(HERE, "results", "niche4_ppmd.json"), "w"), indent=1)

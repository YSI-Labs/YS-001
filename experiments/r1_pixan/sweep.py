"""Barridos SOLO en dev (ultimo 5% de train). Resultados -> sweep_dev.json"""
import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'experiments', 'r1_pixan'))
import devlib, model as M, numpy as np, os
import baseline_wb as WBm
rows = []
def go(m, tag):
    r = devlib.dev_eval(m, n_eval=20000); r['tag'] = tag; rows.append(r)
    print(tag, round(r['bpc'],4), r['n_params'], round(r['fit_wall_s'],1), flush=True)
    devlib.log(rows, os.path.join(devlib.HERE, "sweep_dev.json"))
# referencia WB-5 en el mismo dev
fit_part, dev_arr, V = devlib.dev_split()
r = devlib.dev_eval(WBm.WittenBell(5), n_eval=20000); r['tag'] = "ref_WB5"; rows.append(r); print("ref WB5", round(r['bpc'],4), flush=True)
# A. exactos (sin SimHash)
for k in (1,2,3,4,5):
    go(M.ExactCtx(k), f"exact_k{k}")
# B. b sweep (T=8, K=9, d=4)
for b in (10,12,14,16,18,20):
    go(M.Pixan(T=8, b=b, d=4), f"b{b}_T8")
# C. T sweep (b=16)
for T in (1,2,4,16,32):
    go(M.Pixan(T=T, b=16, d=4), f"T{T}_b16")
# D. dimension d (b=16,T=8)
for d in (2,8,16):
    go(M.Pixan(T=8, b=16, d=d), f"d{d}_b16_T8")
# E. conjuntos de escalas (b=16,T=8,d=4)
sets = {"taus_1-4": [1,2,4], "taus_1-16": [1,2,4,8,16], "taus_16-256": [16,32,64,128,256], "tau_256_only": [256], "tau_1_only": [1]}
for k,v in sets.items():
    go(M.Pixan(T=8, b=16, d=4, taus=v), f"scales_{k}")

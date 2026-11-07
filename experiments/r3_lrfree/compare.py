"""Tabla resumen + test pareado (posicion a posicion) contra el baseline AdamW (r3_gru_paired).

Uso: python -I compare.py tag1 tag2 ...   (lee result_<tag>.json y nll_<tag>.npy de esta carpeta)
SE: error estandar sobre 50 bloques contiguos de posiciones (robusto a autocorrelacion).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(HERE), "r3_gru_paired")
base_nll = np.load(os.path.join(BASE_DIR, "nll_gru_n10000.npy"))
base_bpc = float(json.load(open(os.path.join(BASE_DIR, "result_gru.json")))["bpc"])

rows = []
for tag in sys.argv[1:]:
    res = json.load(open(os.path.join(HERE, f"result_{tag}.json")))
    nll = np.load(os.path.join(HERE, f"nll_{tag}.npy"))
    assert len(nll) == len(base_nll), "mismatch de posiciones de evaluacion"
    diff = nll - base_nll  # >0: el metodo es peor que el baseline
    blocks = np.array([b.mean() for b in np.array_split(diff, 50)])
    se = blocks.std(ddof=1) / np.sqrt(len(blocks))
    rows.append(dict(
        tag=tag, bpc=round(res["bpc"], 4), diff_vs_adamw=round(float(diff.mean()), 4), se_block=round(float(se), 4),
        steps=res["steps"], params=res["n_params"], train_cpu_s=res.get("fit_cpu_s"), fit_wall_s=round(res["fit_wall_s"], 1),
        leak_free=res.get("leak_free"), diverged_at=res.get("diverged_at_step"),
        final_dev=(res["curve_dev"][-1]["dev_bpc"] if res.get("curve_dev") else None),
        dev_curve=[(c["step"], c["dev_bpc"]) for c in res.get("curve_dev", [])],
    ))

print(f"baseline AdamW bpc={base_bpc:.4f} (n_eval=10000)")
for r in rows:
    print(json.dumps(r))
out = os.path.join(HERE, "summary_" + "_".join(sys.argv[1:]) + ".json")
json.dump(dict(baseline_bpc=base_bpc, rows=rows), open(out, "w"), indent=1)
print("saved", out)

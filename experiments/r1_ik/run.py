"""Corrida reproducible de IK (espectral). Ejemplo:
    python -I run.py --r 256 --Dbits 14 --mix dev --n_eval 20000 --tag r256_D14_mix
Escribe results/<tag>.json (metricas de harness + diagnosticos de dev del fit)."""
import os
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
import sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import harness
from model import IKSpectral

ap = argparse.ArgumentParser()
ap.add_argument("--r", type=int, default=256)
ap.add_argument("--Dbits", type=int, default=14)
ap.add_argument("--mix", default="dev", choices=["dev", "pure"])
ap.add_argument("--n_eval", type=int, default=20000)
ap.add_argument("--gen", type=int, default=400)
ap.add_argument("--tag", default=None)
ap.add_argument("--proj", default="svd", choices=["svd", "random"])
a = ap.parse_args()
tag = a.tag or f"r{a.r}_D{a.Dbits}_{a.mix}"

m = IKSpectral(r=a.r, Dbits=a.Dbits, mix=a.mix, proj=a.proj)
out = os.path.join(HERE, "results", f"{tag}.json")
res = harness.run(m, n_eval=a.n_eval, gen_chars=a.gen, out_json=out)
res.update(m.info)
res["tag"] = tag
res["n_params"] = int(m.n_params())   # tras fit: incluye WB solo si alpha<1
res["alpha_final"] = m.alpha
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
harness.summarize(res)
print("fit info:", {k: m.info[k] for k in ("lam_selected", "alpha_selected", "dev_bpc_selected",
                                         "dev_wb_bpc", "sv_top", "sv_r", "C_nnz")})

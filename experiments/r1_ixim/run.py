"""Corrida reproducible de Ixim. Ajusta UNA vez y evalua con el harness comun.
Uso:  python -I run.py --mode random --L 32 --D 2048 --n_eval 20000 --name R0
      python -I run.py --mode ngram --K 4 --D 2048 --n_eval 20000 --name N0
Escribe results/<name>.json (mismo esquema que harness.run + variantes de calibracion).
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, HERE)
import numpy as np
import harness
from model import Ixim

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="random", choices=["random", "ngram"])
ap.add_argument("--L", type=int, default=32)
ap.add_argument("--D", type=int, default=2048)
ap.add_argument("--d0", type=int, default=256)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_eval", type=int, default=20_000)
ap.add_argument("--gen", type=int, default=400)
ap.add_argument("--name", default="R0")
ap.add_argument("--skip_variants", action="store_true")
a = ap.parse_args()

train, val, itos = harness.load()
V = len(itos)
m = Ixim(mode=a.mode, L=a.L, D=a.D, d0=a.d0, K=a.K, seed=a.seed)

t_wall, t_cpu = time.time(), time.process_time()
m.fit(train, V)
fit_wall, fit_cpu = time.time() - t_wall, time.process_time() - t_cpu
chosen = m.calib
print(f"[{m.name}] fit wall={fit_wall:.1f}s cpu={fit_cpu:.1f}s  lam_rel_gcv={m.lam_rel:.3g} "
      f"Dx={m.Dx} calib(dev)={chosen}", flush=True)
print("dev NLL A:", {k: round(v, 4) for k, v in m.dev_nll_A.items()},
      " B:", {k: round(v, 4) for k, v in m.dev_nll_B.items()}, flush=True)

# Variantes de calibracion: todas evaluadas en val SOLO para el reporte; la eleccion es la de dev.
variants = {}
modes = ["softmax", "clip", "newton"]
for cm in modes:
    if a.skip_variants and cm != chosen:
        continue
    m.calib = cm
    r = harness.evaluate(m, val, V, n_eval=a.n_eval, check_leak=(cm == chosen))
    variants[cm] = {k: r[k] for k in r}
    print(f"  calib={cm:8s} bpc(val)={r['bpc']:.4f} leak_free={r.get('leak_free')} "
          f"inf={r['eval_chars_per_s']:.0f} ch/s", flush=True)
m.calib = chosen
res = variants[chosen]

prompt = val[1000:1064]
t_g = time.time()
sample = harness.generate(m, prompt, a.gen, V, seed=0, itos=itos)
print(f"gen {a.gen} chars in {time.time()-t_g:.1f}s", flush=True)

out = dict(res)
out.update(
    name=m.name, bpc=res["bpc"], leak_free=res.get("leak_free"),
    fit_cpu_s=fit_cpu, fit_wall_s=fit_wall, train_chars=len(train),
    train_chars_per_s=len(train) / max(fit_wall, 1e-9),
    n_params=m.n_params(), n_fixed_random=m.n_fixed_random(),
    calib=chosen, calib_variants=variants, fit_info=m.fit_info,
    gcv_curve_per_row=m.gcv_curve, dev_mse_curve=m.dev_mse_curve,
    args=vars(a), n_eval=a.n_eval,
    prompt="".join(itos[i] for i in prompt), sample=sample,
)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", f"{a.name}.json"), "w"), indent=1)
harness.summarize(out)

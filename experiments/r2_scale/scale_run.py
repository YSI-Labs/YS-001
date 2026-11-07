"""Curva de escala por datos: entrena Ixchel v0.1 con un prefijo del train (frac) y evalua en val (n_eval=20000).
Uso: python -I scale_run.py <frac> <variante: A | AWSL>
Escribe results/scale_<var>_<frac>.json (bpc val, params, fit s, pico RAM, bytes de tablas)."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import json, time
import numpy as np
import psutil
import harness
from core import Ixchel, default_chains

frac = float(sys.argv[1])
var = sys.argv[2]
allc = default_chains()
chains = {k: allc[k] for k in var}
gate_chains = tuple(list(chains.keys())[:2])
train, val, itos = harness.load()
V = len(itos)
n = int(len(train) * frac)
m = Ixchel(chains=chains, gate_chains=gate_chains, gated=True, name=f"Ixchel-v0.1-{var}-f{frac}", verbose=False)
t0 = time.time()
m.fit(train[:n], V)
fit_wall = time.time() - t0
peak = psutil.Process().memory_info().peak_wset
tab_bytes = sum(sum(v.nbytes for k, v in t.items() if isinstance(v, np.ndarray)) for t in m.tabs.values())
pairs = m.n_params()
ctxs = sum(len(t["ckey"]) for t in m.tabs.values())
res = harness.evaluate(m, val, V, n_eval=20000)
out = dict(var=var, frac=frac, train_chars=n, bpc=res["bpc"], leak_free=res["leak_free"],
           max_norm_err=res["max_norm_err"], n_params_pairs=pairs, n_contexts=int(ctxs),
           table_bytes=int(tab_bytes), fit_wall_s=fit_wall, peak_rss_mb=peak / 2**20,
           dev_bpc_mix=m.dev_bpc_mix, eval_chars_per_s=res["eval_chars_per_s"], threads=2)
print(json.dumps(out))
json.dump(out, open(os.path.join(HERE, "results", f"scale_{var}_{frac}.json"), "w"), indent=1)

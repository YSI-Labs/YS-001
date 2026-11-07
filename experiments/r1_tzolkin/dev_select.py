"""Seleccion de la regla de combinacion SOLO con dev (ultimo 5% de train). Val no se toca aqui."""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import harness
from model import Tzolkin

train, val, itos = harness.load()
V = len(itos)
n_dev = int(len(train) * 0.05)
fit_part, dev = train[:-n_dev], train[-n_dev:]
out = {}
M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
b = int(sys.argv[2]) if len(sys.argv) > 2 else 16
hash_mode = sys.argv[3] if len(sys.argv) > 3 else "dense"
base = Tzolkin(M=M, b=b, mask="random", mix="lin_wb", hash=hash_mode)
t0 = time.time()
base.fit(fit_part, V)
fit_s = time.time() - t0
print(f"fit (sin dev) M={M} b={b} {hash_mode}: {fit_s:.1f}s  nonzero={base.n_nonzero():,}", flush=True)
for mix in ["lin_wb", "geo_wb", "lin_T", "geo_logT", "geo_seen"]:
    base.mix = mix
    base.name = f"dev-{mix}"
    r = harness.evaluate(base, dev, V, n_eval=20000, check_leak=False)
    out[mix] = dict(bpc_dev=r["bpc"], eval_chars_per_s=r["eval_chars_per_s"], min_prob=r["min_prob"], max_norm_err=r["max_norm_err"])
    print(mix, out[mix], flush=True)
json.dump(dict(M=M, b=b, hash=hash_mode, fit_s_without_dev=fit_s, results=out), open(os.path.join(HERE, f"dev_select_M{M}_b{b}_{hash_mode}.json"), "w"), indent=1)

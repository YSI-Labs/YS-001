"""Confirma en dev (5% final de train, sin tocar val) random vs contig con M=16 exact y regla geo_seen."""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import harness
from model import Tzolkin
train, val, itos = harness.load(); V = len(itos)
n_dev = int(len(train) * 0.05); fit_part, dev = train[:-n_dev], train[-n_dev:]
out = {}
for mask in ["random", "contig"]:
    m = Tzolkin(M=16, b=0, mask=mask, mix="geo_seen", hash="exact")
    m.fit(fit_part, V)
    r = harness.evaluate(m, dev, V, n_eval=20000, check_leak=False)
    out[mask] = dict(bpc_dev=r["bpc"])
    print(mask, out[mask], flush=True)
json.dump(out, open(os.path.join(HERE, "dev_check_contig.json"), "w"), indent=1)

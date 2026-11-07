"""Reproduce la base v0.1 (Ixchel default, cadenas A,W,S,L, compuerta A+W) con el harness, n_eval=20000."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import harness
from core import Ixchel
if __name__ == "__main__":
    m = Ixchel(name="Ixchel-v0.1-base-repro", verbose=True)
    res = harness.run(m, n_eval=20_000, gen_chars=400, out_json=os.path.join(HERE, "base_repro.json"))
    res.update(dev_bpc_chains=m.dev_bpc_chains, dev_bpc_mix=m.dev_bpc_mix)
    import json; json.dump(res, open(os.path.join(HERE, "base_repro.json"), "w"), indent=1)
    harness.summarize(res)

"""Seleccion de configuracion SOLO con un dev interno (ultimo 5% de train). No mira val."""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import harness
from model import Yaxche

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    tag = sys.argv[2] if len(sys.argv) > 2 else "dev"
    train, val, itos = harness.load()
    V = len(itos)
    cut = int(len(train) * 0.95)
    tr, dev = train[:cut], train[cut:]
    m = Yaxche(itos, **kw)
    t0 = time.time()
    m.fit(tr, V)
    fit = time.time() - t0
    out = {"kw": kw, "fit_wall_s": fit, "per_tree": m.tree_info}
    Bs = [b for b in (1, 4, 16, 64) if b <= m.B]
    for b in Bs:
        m.set_B(b)
        r = harness.evaluate(m, dev, V, n_eval=20_000, check_leak=False)
        out[f"B{b}"] = dict(bpc=r["bpc"], n_params=m.n_params(), eval_s=r["eval_time_s"],
                            fit_s_prefix=float(sum(m.fit_s_per_tree[:b])) + m.F_build_s)
        print(tag, b, round(r["bpc"], 4), m.n_params(), flush=True)
    print(tag, "fit_wall", round(fit, 1), "leaves", [t["leaves"] for t in m.tree_info][:4],
          "maxdepth", [t["max_leaf_depth"] for t in m.tree_info][:4], flush=True)
    json.dump(out, open(os.path.join(HERE, f"dev_{tag}.json"), "w"), indent=1)

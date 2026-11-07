"""Evaluacion final con error estandar. Ajusta cada config una vez y evalua en n_eval=10000 (mismas posiciones que la GRU)
y 50000, con nll por posicion -> media, error estandar (sd/sqrt(n)) y bootstrap por bloques (bloques de 500 posiciones)."""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness
import run_v02
from core_v02 import Ixchel

def nll_at(model, val, V, n_eval, batch=4096):
    pos = harness.eval_positions(val, n_eval)
    out = np.zeros(len(pos))
    for s in range(0, len(pos), batch):
        pp = pos[s:s+batch]
        P = model.predict(val, pp).astype(np.float64)
        out[s:s+len(pp)] = -np.log2(np.maximum(P[np.arange(len(pp)), val[pp]], 1e-12))
    return out

def stats(x, block=500):
    n = len(x); nb = n // block
    bm = x[:nb*block].reshape(nb, block).mean(1)
    return dict(bpc=float(x.mean()), se_iid=float(x.std(ddof=1)/np.sqrt(n)), se_block=float(bm.std(ddof=1)/np.sqrt(nb)), n=n)

if __name__ == "__main__":
    train, val, itos = harness.load(); V = len(itos)
    res = {}
    for c in sys.argv[1:]:
        m = Ixchel(name=f"Ixchel-{c}", verbose=False, **run_v02.CFG[c])
        t0 = time.time(); m.fit(train, V); fit = time.time() - t0
        r = dict(fit_wall_s=fit, n_params=m.n_params())
        for ne in (10_000, 50_000):
            t1 = time.time(); x = nll_at(m, val, V, ne)
            r[f"eval{ne}"] = stats(x); r[f"eval{ne}"]["eval_s"] = time.time() - t1
        res[c] = r
        print(c, json.dumps(r), flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "final_eval.json"), "w"), indent=1)

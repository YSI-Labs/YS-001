"""Ixchel v0.4 = v0.3 + cache de historial de orden 8 (KeyedWin, r3_longcache) en lugar del CacheWB de orden 4 + PoE con Hessiano BLAS.
Uso: python -I run_v04.py <config> [n_eval]   config: v04 | v04_c128 (ventana 128) | v04_nocache"""
import os, sys, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
from core_v04 import Ixchel, default_chains
from longcache import KeyedWin
INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
def cfg(name, chains=None):
    ch = chains if chains is not None else {**default_chains(), **LEX}
    if name == "v04":
        xp = (KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),)
    elif name == "v04_c128":
        xp = (KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[128]),)
    elif name == "v04_nocache":
        xp = ()
    else: raise SystemExit("config?")
    return dict(chains=ch, mix="poe", cache_windows=(), xperts=xp)
if __name__ == "__main__":
    c = sys.argv[1]; n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    m = Ixchel(name=f"Ixchel-v0.4-{c}", **cfg(c))
    outp = os.path.join(HERE, "results", f"v04_{c}.json")
    res = harness.run(m, n_eval=n_eval, gen_chars=500, out_json=outp)
    res.update(dev_bpc_mix=m.dev_bpc_mix, poe_w=None if m.poe_w is None else m.poe_w.tolist())
    json.dump(res, open(outp, "w"), indent=1); harness.summarize(res)

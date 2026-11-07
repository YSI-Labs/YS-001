"""Ixchel v0.2: v0.1 + cache en contexto + mezcla geometrica global. Uso: python -I run_v02.py <config> [n_eval]
Configs: poe | cache_em | cache_poe | cache2_poe"""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
from core_v02 import Ixchel, default_chains
INF = 1 << 40
CFG = {
    "poe":        dict(cache_windows=(), mix="poe"),
    "cache_em":   dict(cache_windows=(256, 1024, 4096, INF), mix="em"),
    "cache_poe":  dict(cache_windows=(256, 1024, 4096, INF), mix="poe"),
    "cache2_poe": dict(cache_windows=(1024, INF), mix="poe"),
    "cacheB_poe": dict(cache_windows=(256, 1024), mix="poe"),     # ventanas acotadas (comparable a ctx neuronal ~128-1024)
    "cache128_poe": dict(cache_windows=(128,), mix="poe"),        # solo 128 chars: mismo contexto que la GRU/transformer
}
LEX = dict(
    Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
    Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
    WL=[[], ["wlen"], ["wlen", "c1", "c2"]],
    W2=[[], ["wp"], ["wp", "pw2"]])
CFG["v03"] = dict(chains={**default_chains(), **LEX}, cache_windows=(256, 1024, 4096, INF), mix="poe")
CFG["v03_c128"] = dict(chains={**default_chains(), **LEX}, cache_windows=(128,), mix="poe")
CFG["v03_nocache"] = dict(chains={**default_chains(), **LEX}, cache_windows=(), mix="poe")
if __name__ == "__main__":
    c = sys.argv[1]; n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    m = Ixchel(name=f"Ixchel-v0.2-{c}", **CFG[c])
    outp = os.path.join(HERE, "results", f"v02_{c}.json")
    res = harness.run(m, n_eval=n_eval, gen_chars=500, out_json=outp)
    res.update(dev_bpc_mix=m.dev_bpc_mix, dev_bpc_experts=getattr(m, "dev_bpc_experts", None),
               poe_w=None if m.poe_w is None else m.poe_w.tolist(), expert_names=m.expert_names())
    json.dump(res, open(outp, "w"), indent=1)
    harness.summarize(res)

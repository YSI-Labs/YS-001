"""Corre variantes de Ixchel v0.1 con el harness comun. Uso: python -I run_v01.py <variante> [n_eval]
Variantes: A (solo cadena contigua KN), AW, AWS, AWSL (todas), AWSL_nogate (pesos globales)."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
from core import Ixchel, default_chains

if __name__ == "__main__":
    var = sys.argv[1] if len(sys.argv) > 1 else "A"
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    allc = default_chains()
    sel = var.split("_")[0]
    chains = {k: allc[k] for k in sel}
    gated = not var.endswith("nogate")
    gate_chains = tuple(list(chains.keys())[:2])
    m = Ixchel(chains=chains, gate_chains=gate_chains, gated=gated, name=f"Ixchel-v0.1-{var}")
    outp = os.path.join(HERE, "results", f"v01_{var}.json")
    res = harness.run(m, n_eval=n_eval, gen_chars=500, out_json=outp)
    res.update(dev_bpc_chains=m.dev_bpc_chains, dev_bpc_mix=m.dev_bpc_mix)
    import json; json.dump(res, open(outp, "w"), indent=1)
    harness.summarize(res)

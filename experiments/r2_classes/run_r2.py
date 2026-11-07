"""Corre variantes R2 (classes) con el harness comun. Uso: python -I run_r2.py <variante> [n_eval]
Variantes:
  base        Ixchel v0.1 AWSL (control; debe dar 2.078)
  pcA/pcB/pcC cadenas de clusters de palabra (pc1,pc2) anadidas a AWSL
  cAuto       clases de caracter elegidas por BIC (sin cluster de palabra)
  c4/c8/c16/c32  clases de caracter con k fijo (sin cluster de palabra)
  combo_*     combinaciones (se definen tras la iteracion 1)
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
import core
from core import Ixchel, default_chains, build_char_tables
from feats import R2Prov

CLUSTER_CHAINS = {
    "pcA": [[], ["pc1"], ["pc1", "pc2"]],
    "pcB": [[], ["pc1"], ["pc1", "pc2"], ["wp", "pc1", "pc2"], ["wp", "pc1", "pc2", "c1"]],
    "pcC": [[], ["pc1"], ["pc1", "pc2"], ["pc1", "pc2", "c1", "c2"]],
}


def variant(name):
    allc = default_chains()
    chains = {k: allc[k] for k in "AWSL"}
    prov_kw = None  # None = sin proveedor
    if name == "base":
        pass
    elif name in CLUSTER_CHAINS:
        chains["C"] = CLUSTER_CHAINS[name]
        prov_kw = dict(use_words=True)
    elif name == "cAuto" or name.startswith("c") and name[1:].isdigit():
        k = "auto" if name == "cAuto" else int(name[1:])
        prov_kw = dict(use_words=False, char_k=k)
    elif name.startswith("combo_"):
        # combo_<pcX>_<k|auto>
        _, pc, ck = name.split("_")
        chains["C"] = CLUSTER_CHAINS[pc]
        prov_kw = dict(use_words=True, char_k=("auto" if ck == "auto" else int(ck)))
    else:
        raise SystemExit(f"variante desconocida {name}")
    return chains, prov_kw


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "base"
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    chains, prov_kw = variant(name)
    prov = None
    if prov_kw is not None:
        itos = json.load(open(os.path.join(core.DATA, "vocab.json")))
        cls0, isl, isnl = build_char_tables(itos)
        prov = R2Prov(default_cls=cls0, isletter=isl, **prov_kw)
    m = Ixchel(chains=chains, gate_chains=("A", "W"), gated=True, name=f"R2-{name}", prov=prov, verbose=False)
    outp = os.path.join(HERE, "results", f"r2_{name}_n{n_eval}.json")
    res = harness.run(m, n_eval=n_eval, gen_chars=500, out_json=None)
    res.update(dev_bpc_chains=m.dev_bpc_chains, dev_bpc_mix=m.dev_bpc_mix,
               params_by_chain={nm: int(len(m.tabs[nm]["psym"])) for nm in m.names},
               prov_info=(prov.info if prov is not None else {}),
               prov_time_s=getattr(m, "prov_time", 0.0), variant=name)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), indent=1)
    harness.summarize(res)
    print("dev chains:", {k: round(v, 4) for k, v in m.dev_bpc_chains.items()}, "dev mix:", round(m.dev_bpc_mix, 4))
    print("prov_info:", res["prov_info"], "prov_time_s:", round(res["prov_time_s"], 1))

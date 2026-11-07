"""r4_codeatoms: plantilla run_mix15 + 6 cadenas de CODIGO con atomos nuevos (36..44).
Uso: python -I run_codeatoms.py <variante>
  base          : sin cadenas nuevas (control; debe reproducir ixchel/results/mix15_base.json)
  full          : base + 6 cadenas de codigo
  drop:<atomo>  : full sin ese atomo (ablacion por atomo; las cadenas se deduplican)
Salida: results/mix15_<variante>.json y results/mix15_<variante>_nll.npz (nll por dominio, pareado vs base)."""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "mix15"; os.environ.setdefault("NUMBA_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "ixchel")); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness, eval_suite as ES
import psutil
from core_codeatoms import Ixchel, default_chains
from longcache import KeyedWin
INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
# Cadenas de CODIGO (niveles de grueso a fino; cada nivel es subconjunto del siguiente => KN valido)
CODE = dict(
    CK=[[], ["indp"], ["indp", "kwl"], ["indp", "kwl", "c1"]],                      # indent. previa, ult. keyword, c1
    CB=[[], ["bdep"], ["bdep", "ind"], ["bdep", "ind", "c1", "c2"]],               # profundidad, indent., c1 c2
    CI=[[], ["idl"], ["idl", "c1"], ["idl", "idp", "c1"]],                         # ultimo id, penultimo id (x = ...)
    CF=[[], ["fwp"], ["fwp", "ind"], ["fwp", "ind", "wp"]],                        # primer token previa, indent., prefijo palabra
    CS=[[], ["strst"], ["strst", "c1"], ["strst", "c1", "c2"], ["strst", "endc", "c1", "c2", "c3"]],  # cadena/comentario, c1..c3
    CSK=[[], ["ind", "c1"], ["ind", "c1", "c3"], ["ind", "c1", "c3", "c4"]],       # contextos saltados que conservan c1
)
NEW_ATOMS = ["ind", "indp", "bdep", "kwl", "fwp", "strst", "idl", "idp", "endc"]


def drop_atom(levels, atom):
    out = []
    for lvl in levels:
        l2 = [a for a in lvl if a != atom]
        if not out or out[-1] != l2:
            out.append(l2)
    return out


def build(extra_chains):
    ch = {**default_chains(), **LEX, **extra_chains}
    return Ixchel(name="codeatoms", verbose=True, chains=ch, mix="poe", cache_windows=(),
                  xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))


def variant_chains(variant):
    if variant == "base":
        return {}
    if variant == "full":
        return dict(CODE)
    if variant == "prune":
        # atomos que las ablaciones individuales no justifican (ind, indp, kwl, idp) + los que no cambian nada (strst, endc)
        out = {}
        for k, lv in CODE.items():
            lv2 = lv
            for a in ("ind", "indp", "kwl", "idp", "strst", "endc"):
                lv2 = drop_atom(lv2, a)
            if len(lv2) > 1:
                out[k] = lv2
        return out
    if variant.startswith("drop:"):
        atom = variant.split(":", 1)[1]
        assert atom in NEW_ATOMS, atom
        out = {}
        for k, lv in CODE.items():
            lv2 = drop_atom(lv, atom)
            if len(lv2) > 1:
                out[k] = lv2
        return out
    raise ValueError(variant)


if __name__ == "__main__":
    variant = sys.argv[1]
    tag = variant.replace(":", "_")
    out_dir = os.path.join(HERE, "results")
    D = os.path.join(ROOT, "data", "mix15")
    chains = variant_chains(variant)
    train, val, itos = harness.load(); V = len(itos)
    m = build(chains)
    proc = psutil.Process()
    t = time.time(); m.fit(train, V); fit_s = time.time() - t
    peak_mb = proc.memory_info().rss / 2**20
    print("fit", round(fit_s), "s  rss", round(peak_mb), "MB", flush=True)
    dom = ES.domain_bpc(m, D, V, n_eval=30_000)
    nll = {d: dom[d].pop("nll") for d in dom}
    np.savez(os.path.join(out_dir, f"mix15_{tag}_nll.npz"), **nll)
    names = m.expert_names()
    out = dict(tag=tag, variant=variant, chains={k: v for k, v in chains.items()}, n_params=int(m.n_params()),
               dev_bpc_mix=m.dev_bpc_mix, fit_s=fit_s, rss_mb_after_fit=peak_mb,
               poe_w={n: float(w) for n, w in zip(names, m.poe_w)}, dev_bpc_experts=m.dev_bpc_experts,
               domain_bpc=dom)
    print(tag, "bpc", {d: round(v["bpc"], 4) for d, v in dom.items()}, "params", out["n_params"], flush=True)
    json.dump(out, open(os.path.join(out_dir, f"mix15_{tag}.json"), "w"), indent=1)

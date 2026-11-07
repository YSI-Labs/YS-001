"""Plantilla de experimentos RAPIDOS sobre mix15 (15M chars de entrenamiento, ~5 min por ajuste). Uso: python -I run_mix15.py <tag>
Evalua: bpc por dominio (30k pos c/u), HellaSwag n=1500, PIQA n=800, ARC-Easy n=1000 y guarda nll por dominio para tests PAREADOS
(results/mix15_<tag>.json y mix15_<tag>_nll.npz). Copiala y modifica la configuracion en tu carpeta."""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "mix15"; os.environ.setdefault("NUMBA_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness, eval_suite as ES
from core_v04 import Ixchel, default_chains
from longcache import KeyedWin
INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
def base_model(**kw):
    ch = {**default_chains(), **LEX}
    return Ixchel(name="mix15", verbose=True, chains=ch, mix="poe", cache_windows=(),
                  xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),), **kw)
def evaluate(m, tag, out_dir):
    train, val, itos = harness.load(); V = len(itos); enc = ES.make_encoder(itos); D = os.path.join(ROOT, "data", "mix15")
    out = dict(tag=tag, n_params=int(m.n_params()), dev_bpc_mix=m.dev_bpc_mix)
    dom = ES.domain_bpc(m, D, V, n_eval=30_000); nll = {d: dom[d].pop("nll") for d in dom}; out["domain_bpc"] = dom
    np.savez(os.path.join(out_dir, f"mix15_{tag}_nll.npz"), **nll)
    print(tag, "bpc", {d: (round(v["bpc"], 3), round(v["top1_acc"], 3)) for d, v in dom.items()}, flush=True)
    out["hellaswag"] = ES.hellaswag(m, enc, V, n=1500); out["piqa"] = ES.piqa(m, enc, V, n=800); out["arc_easy"] = ES.arc_easy(m, enc, V, n=1000)
    print(tag, {k: round(100 * out[k]["acc_norm"]["acc"], 1) for k in ("hellaswag", "piqa", "arc_easy")}, flush=True)
    json.dump(out, open(os.path.join(out_dir, f"mix15_{tag}.json"), "w"), indent=1); return out
if __name__ == "__main__":
    tag = sys.argv[1]; train, val, itos = harness.load(); V = len(itos)
    m = base_model(); t = time.time(); m.fit(train, V); print("fit", round(time.time() - t), "s", flush=True)
    evaluate(m, tag, os.path.join(HERE, "results"))

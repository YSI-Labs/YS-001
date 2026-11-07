"""r4_codeatoms: HumanEval honesto (eval_suite.humaneval, greedy, 164 problemas) + validez sintactica de codigo generado
(eval_suite.code_syntax_validity) sobre UNA variante. Reajusta el modelo (misma config que run_codeatoms.py).
Uso: python -I eval_best.py <variante>   (p.ej. full | base)   -> results/he_<variante>.json"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "mix15"; os.environ.setdefault("NUMBA_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "ixchel")); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness, eval_suite as ES
from run_codeatoms import build, variant_chains

if __name__ == "__main__":
    variant = sys.argv[1]; tag = variant.replace(":", "_")
    D = os.path.join(ROOT, "data", "mix15")
    train, val, itos = harness.load(); V = len(itos); enc = ES.make_encoder(itos)
    m = build(variant_chains(variant)); t = time.time(); m.fit(train, V); print("fit", round(time.time() - t), "s", flush=True)
    t = time.time()
    he = ES.humaneval(m, enc, V, itos, max_chars=350, keep_ctx=1500)
    print("humaneval", {k: v for k, v in he.items() if k != "samples"}, round(time.time() - t), "s", flush=True)
    t = time.time()
    syn = ES.code_syntax_validity(m, enc, V, itos, D, n_prompts=120, prefix_chars=600, gen_chars=250, keep_ctx=1500)
    print("code_syntax_validity", syn, round(time.time() - t), "s", flush=True)
    json.dump(dict(tag=tag, variant=variant, humaneval=he, code_syntax=syn),
              open(os.path.join(HERE, "results", f"he_{tag}.json"), "w"), indent=1, ensure_ascii=False)

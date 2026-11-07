"""Entrena Ixchel en mix1 y corre la bateria completa. Uso: python -I run_mix_full.py <tag> [n_train|0=todo] [quick]
Guarda results/mix_<tag>.json (+ nll por dominio en results/mix_<tag>_nll.npz)."""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "mix1"; os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness, eval_suite as ES
from core_v02 import Ixchel, default_chains
INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
def build(variant):
    ch = {**default_chains(), **LEX}
    if variant == "base":
        return dict(chains=ch, mix="poe", cache_windows=(256, 1024, 4096, INF))
    raise SystemExit("variant?")
if __name__ == "__main__":
    tag = sys.argv[1]; n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 0; quick = len(sys.argv) > 3 and sys.argv[3] == "quick"
    train, val, itos = harness.load(); V = len(itos); enc = ES.make_encoder(itos)
    if n_train: train = train[:n_train]
    m = Ixchel(name=f"Ixchel-mix-{tag}", verbose=True, **build("base"))
    t0 = time.time(); c0 = time.process_time(); m.fit(train, V)
    out = dict(tag=tag, n_train=int(len(train)), n_params=int(m.n_params()), fit_wall_s=time.time() - t0, fit_cpu_s=time.process_time() - c0,
               poe_w=None if m.poe_w is None else dict(zip(m.expert_names(), [float(x) for x in m.poe_w])), dev_bpc_mix=m.dev_bpc_mix)
    try:
        import psutil; out["peak_rss_mb"] = psutil.Process().memory_info().peak_wset / 2**20
    except Exception: pass
    D = os.path.join(ROOT, "data", "mix1")
    print("== bpc por dominio", flush=True)
    t1 = time.time(); dom = ES.domain_bpc(m, D, V, n_eval=3000 if quick else 30_000)
    nll = {d: dom[d].pop("nll") for d in dom}; out["domain_bpc"] = dom
    np.savez(os.path.join(HERE, "results", f"mix_{tag}_nll.npz"), **nll)
    print({d: round(v["bpc"], 3) for d, v in dom.items()}, f"({time.time()-t1:.0f}s)", flush=True)
    nq = 150 if quick else None
    print("== opcion multiple", flush=True)
    t1 = time.time()
    out["hellaswag"] = ES.hellaswag(m, enc, V, n=(200 if quick else 5000)); print("hellaswag", out["hellaswag"], f"({time.time()-t1:.0f}s)", flush=True)
    out["piqa"] = ES.piqa(m, enc, V, n=(150 if quick else 1838)); print("piqa", out["piqa"], flush=True)
    out["arc_easy"] = ES.arc_easy(m, enc, V, n=(150 if quick else 2376)); print("arc_easy", out["arc_easy"], flush=True)
    print("== humaneval", flush=True); t1 = time.time()
    out["humaneval"] = ES.humaneval(m, enc, V, itos, limit=(8 if quick else None)); print("humaneval", {k: v for k, v in out["humaneval"].items() if k != "samples"}, f"({time.time()-t1:.0f}s)", flush=True)
    print("== validez sintactica", flush=True); t1 = time.time()
    out["code_syntax"] = ES.code_syntax_validity(m, enc, V, itos, D, n_prompts=(8 if quick else 120)); print(out["code_syntax"], f"({time.time()-t1:.0f}s)", flush=True)
    samples = {}
    for name, prompt in [("web", "The history of the city begins"), ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
                         ("math", "Question: A farmer has 12 cows and buys 5 more. Each cow produces 3 liters of milk per day. How many liters in total per day?\nAnswer:"),
                         ("chat", "### Instruction:\nGive me three tips to learn to cook.\n\n### Response:\n"), ("books", "HAMLET.\nTo be, or not to be, ")]:
        samples[name] = ES.generate_greedy(m, enc, V, prompt, 350, itos, temperature=0.8, rng=np.random.default_rng(0), keep_ctx=1500)
    out["samples"] = samples
    json.dump(out, open(os.path.join(HERE, "results", f"mix_{tag}.json"), "w"), indent=1)
    print("DONE", json.dumps({k: out[k] for k in ("n_train", "n_params", "fit_wall_s")}), flush=True)

"""Fase final del modelo grande: carga las tablas construidas (mmap), ajusta los pesos PoE en dev, mide costos y corre la bateria.
Uso: python -I run_big_eval.py <tag> [quick]"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "big1"; os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness, eval_suite as ES
from core_v04 import Ixchel, default_chains
from longcache import KeyedWin
from bigbuild import load_tables
from run_bigbuild import LEX
INF = 1 << 40
if __name__ == "__main__":
    tag = sys.argv[1]; quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
    D = os.path.join(ROOT, "data", "big1")
    train = np.load(os.path.join(D, "train.npy"), mmap_mode="r"); n = len(train); cut = int(n * 0.95)
    itos = json.load(open(os.path.join(D, "vocab.json"))); V = len(itos); enc = ES.make_encoder(itos)
    ch = {**default_chains(), **LEX}
    m = Ixchel(name=f"Ixchel-big-{tag}", verbose=True, chains=ch, mix="poe", cache_windows=(),
               xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))
    m.dev_span = 10_000_000; m.dev_positions = 150_000
    t = time.time(); m.prebuilt_tabs = load_tables(os.path.join(ROOT, "data", "big1_model"), m.names, mmap=True); t_load = time.time() - t
    cnt = np.zeros(V, dtype=np.float64)
    for i in range(0, cut, 50_000_000): cnt += np.bincount(np.asarray(train[i:min(i + 50_000_000, cut)]), minlength=V)
    m.base_override = cnt / cnt.sum()
    t = time.time(); m.fit(train, V); t_fit = time.time() - t
    pairs = {nm: int(len(tb["psym"])) for nm, tb in m.tabs.items()}
    out = dict(tag=tag, n_train_chars=int(n), n_built_chars=int(cut), n_params=int(sum(pairs.values())), pairs_per_chain=pairs, load_tables_s=t_load, dev_fit_s=t_fit,
               poe_w=dict(zip(m.expert_names(), [float(x) for x in m.poe_w])), dev_bpc_mix=m.dev_bpc_mix,
               table_bytes=int(sum(os.path.getsize(os.path.join(ROOT, "data", "big1_model", nm, f)) for nm in m.names for f in os.listdir(os.path.join(ROOT, "data", "big1_model", nm)))))
    json.dump(out, open(os.path.join(HERE, "results", f"big_{tag}.json"), "w"), indent=1); print("PARAMS", out["n_params"], "table_GB", round(out["table_bytes"] / 2**30, 2), flush=True)
    # ---- latencia / throughput de inferencia con tablas en disco/cache de paginas
    val = np.load(os.path.join(D, "val_web.npy")); pos = harness.eval_positions(val, 4096)
    tt = []
    for _ in range(3):
        t = time.time(); m.predict(val, pos); tt.append(time.time() - t)
    out["predict_batch4096_s"] = tt; out["predict_pos_per_s"] = float(4096 / np.median(tt))
    t = time.time(); [m.predict(val[:3000], np.array([2999 - i], dtype=np.int64)) for i in range(50)]; out["predict_single_ms"] = (time.time() - t) / 50 * 1000
    print("inferencia", out["predict_pos_per_s"], "pos/s ;", out["predict_single_ms"], "ms/pos individual", flush=True)
    dom = ES.domain_bpc(m, D, V, n_eval=3000 if quick else 30_000); nll = {d: dom[d].pop("nll") for d in dom}
    np.savez(os.path.join(HERE, "results", f"big_{tag}_nll.npz"), **nll); out["domain_bpc"] = dom
    print("bpc", {d: (round(v["bpc"], 3), round(v["top1_acc"], 3)) for d, v in dom.items()}, flush=True)
    out["hellaswag"] = ES.hellaswag(m, enc, V, n=(200 if quick else 5000)); print("hellaswag", out["hellaswag"], flush=True)
    out["piqa"] = ES.piqa(m, enc, V, n=(150 if quick else 1838)); print("piqa", out["piqa"], flush=True)
    out["arc_easy"] = ES.arc_easy(m, enc, V, n=(150 if quick else 2376)); print("arc_easy", out["arc_easy"], flush=True)
    out["humaneval"] = ES.humaneval(m, enc, V, itos, limit=(8 if quick else None)); print("humaneval", {k: v for k, v in out["humaneval"].items() if k != "samples"}, flush=True)
    out["code_syntax"] = ES.code_syntax_validity(m, enc, V, itos, D, n_prompts=(8 if quick else 120)); print("code_syntax", out["code_syntax"], flush=True)
    SAMPLE_PROMPTS = [("web", "The history of the city begins"),
        ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
        ("math", "Question: A farmer has 12 cows and buys 5 more. Each cow produces 3 liters of milk per day. How many liters in total per day?\nAnswer:"),
        ("chat", "### Instruction:\nGive me three tips to learn to cook.\n\n### Response:\n"), ("books", "HAMLET.\nTo be, or not to be, ")]
    out["samples_T1"] = {nm: ES.generate_greedy(m, enc, V, p, 350, itos, temperature=1.0, rng=np.random.default_rng(0), keep_ctx=1500) for nm, p in SAMPLE_PROMPTS}
    json.dump(out, open(os.path.join(HERE, "results", f"big_{tag}.json"), "w"), indent=1); print("BIGEVAL_DONE", flush=True)

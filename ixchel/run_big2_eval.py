"""Fase final del modelo grande v0.5 (big2): tablas mmap + pesos PoE en dev + costos + bateria completa. Uso: python -I run_big2_eval.py <tag>"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "big2"; os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness, eval_suite as ES
from core_v05 import Ixchel
from chains_v05 import v05_chains
from longcache import KeyedWin
from bigbuild import load_tables
INF = 1 << 40
if __name__ == "__main__":
    tag = sys.argv[1]; D = os.path.join(ROOT, "data", "big2")
    train = np.load(os.path.join(D, "train.npy"), mmap_mode="r"); man = json.load(open(os.path.join(D, "manifest.json"))); cut = int(man["main_chars_with_repetition"]); n = len(train)
    itos = json.load(open(os.path.join(D, "vocab.json"))); V = len(itos); enc = ES.make_encoder(itos)
    m = Ixchel(name=f"Ixchel-v05-big2-{tag}", verbose=True, chains=v05_chains(), mix="poe", cache_windows=(), dev_frac=1 - cut / n,
               xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))
    m.dev_span = 10_000_000; m.dev_positions = 150_000
    t = time.time(); m.prebuilt_tabs = load_tables(os.path.join(ROOT, "data", "big2_model"), m.names, mmap=True); t_load = time.time() - t
    cnt = np.zeros(V)
    for i in range(0, cut, 50_000_000): cnt += np.bincount(np.asarray(train[i:min(i + 50_000_000, cut)]), minlength=V)
    m.base_override = cnt / cnt.sum()
    t = time.time(); m.fit(train, V); t_fit = time.time() - t
    pairs = {nm: int(len(tb["psym"])) for nm, tb in m.tabs.items()}
    out = dict(tag=tag, n_params=int(sum(pairs.values())), pairs_per_chain=pairs, n_main_chars=cut, load_tables_s=t_load, dev_fit_s=t_fit,
               poe_w=dict(zip(m.expert_names(), [float(x) for x in m.poe_w])), dev_bpc_mix=m.dev_bpc_mix,
               table_bytes=int(sum(os.path.getsize(os.path.join(ROOT, "data", "big2_model", nm, f)) for nm in m.names for f in os.listdir(os.path.join(ROOT, "data", "big2_model", nm)))))
    json.dump(out, open(os.path.join(HERE, "results", f"big2_{tag}.json"), "w"), indent=1); print("PARAMS", out["n_params"], "GB", round(out["table_bytes"] / 2**30, 2), flush=True)
    val = np.load(os.path.join(D, "val_web.npy")); pos = harness.eval_positions(val, 4096); tt = []
    for _ in range(3): t = time.time(); m.predict(val, pos); tt.append(time.time() - t)
    out["predict_pos_per_s"] = float(4096 / np.median(tt))
    t = time.time(); [m.predict(val[:3000], np.array([2999 - i], dtype=np.int64)) for i in range(50)]; out["predict_single_ms"] = (time.time() - t) / 50 * 1000
    print("inferencia", out["predict_pos_per_s"], "pos/s;", out["predict_single_ms"], "ms/pos", flush=True)
    dom = ES.domain_bpc(m, D, V, n_eval=30_000); nll = {d: dom[d].pop("nll") for d in dom}
    np.savez(os.path.join(HERE, "results", f"big2_{tag}_nll.npz"), **nll); out["domain_bpc"] = dom
    print("bpc", {d: (round(v["bpc"], 3), round(v["top1_acc"], 3)) for d, v in dom.items()}, flush=True)
    out["hellaswag"] = ES.hellaswag(m, enc, V, n=5000); print("hellaswag", out["hellaswag"], flush=True)
    out["piqa"] = ES.piqa(m, enc, V, n=1838); print("piqa", out["piqa"], flush=True)
    out["arc_easy"] = ES.arc_easy(m, enc, V, n=2376); print("arc_easy", out["arc_easy"], flush=True)
    out["humaneval"] = ES.humaneval(m, enc, V, itos); print("humaneval", {k: v for k, v in out["humaneval"].items() if k != "samples"}, flush=True)
    out["code_syntax"] = ES.code_syntax_validity(m, enc, V, itos, D, n_prompts=120); print("code_syntax", out["code_syntax"], flush=True)
    json.dump(out, open(os.path.join(HERE, "results", f"big2_{tag}.json"), "w"), indent=1); print("BIG2EVAL_DONE", flush=True)

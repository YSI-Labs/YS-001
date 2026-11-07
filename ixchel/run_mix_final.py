"""Corrida final sobre mix1: Ixchel v0.4 (todo el train 85.6M) + bateria; luego ENSAMBLE Ixchel x GRU (producto de expertos, 2 pesos
ajustados en dev alineado: ultimo 5% del train, que la GRU tampoco vio) + bateria. Uso: python -I run_mix_final.py <tag> [gru_steps]"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "mix1"; os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
R3G = os.path.join(ROOT, "experiments", "r3_gru_mix")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, torch, importlib.util, harness, eval_suite as ES
from core_v04 import Ixchel, default_chains
from longcache import KeyedWin
from mixing import poe_global
torch.set_num_threads(4)
INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
SAMPLE_PROMPTS = [("web", "The history of the city begins"), ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("math", "Question: A farmer has 12 cows and buys 5 more. Each cow produces 3 liters of milk per day. How many liters in total per day?\nAnswer:"),
    ("chat", "### Instruction:\nGive me three tips to learn to cook.\n\n### Response:\n"), ("books", "HAMLET.\nTo be, or not to be, ")]

class GRUWrap:
    def __init__(self, V, weights):
        spec = importlib.util.spec_from_file_location("gru_model", os.path.join(R3G, "model.py")); gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)
        g = gm.NeuralLM("gru", budget_s=1e9, max_steps=1); g.V = V; g.net = g._build(V)
        g.net.load_state_dict(torch.load(weights)); g.net.eval(); self.m = g; self.V = V
    def predict(self, ids, pos): return self.m.predict(ids, pos)
    @torch.no_grad()
    def seq_logp(self, ids):      # log-prob (nats) del simbolo t+1 dado <=t, para t=0..T-2 : [T-1, V]
        x = torch.from_numpy(np.asarray(ids[:-1], dtype=np.int64))[None, :]
        return torch.log_softmax(self.m.net(x)[0].float(), -1).numpy()

class Ensemble:
    def __init__(self, ix, gru, w): self.ix, self.gru, self.w, self.V = ix, gru, w, ix.V
    def predict(self, ids, pos):
        L = self.w[0] * np.log(np.maximum(self.ix.predict(ids, pos).astype(np.float64), 1e-12)) + self.w[1] * np.log(np.maximum(self.gru.predict(ids, pos).astype(np.float64), 1e-12))
        L -= L.max(1, keepdims=True); P = np.exp(L); return (P / P.sum(1, keepdims=True)).astype(np.float32)
    def score_sequence(self, ids, start):
        pos = np.arange(start, len(ids), dtype=np.int64)
        Li = np.log(np.maximum(self.ix.predict(ids, pos).astype(np.float64), 1e-12))
        Lg = self.gru.seq_logp(ids)[start - 1:].astype(np.float64)
        L = self.w[0] * Li + self.w[1] * Lg
        L -= L.max(1, keepdims=True); L -= np.log(np.exp(L).sum(1, keepdims=True))
        return float(L[np.arange(len(pos)), ids[pos]].sum() / np.log(2))

def battery(name, model, enc, itos, V, D, quick, out, keep_ctx=1500):
    t1 = time.time(); dom = ES.domain_bpc(model, D, V, n_eval=3000 if quick else 30_000)
    nll = {d: dom[d].pop("nll") for d in dom}; np.savez(os.path.join(HERE, "results", f"mixfinal_{name}_nll.npz"), **nll)
    out["domain_bpc"] = dom; print(name, "bpc", {d: (round(v["bpc"], 3), round(v["top1_acc"], 3)) for d, v in dom.items()}, f"({time.time()-t1:.0f}s)", flush=True)
    out["hellaswag"] = ES.hellaswag(model, enc, V, n=(200 if quick else 5000)); print(name, "hellaswag", out["hellaswag"], flush=True)
    out["piqa"] = ES.piqa(model, enc, V, n=(150 if quick else 1838)); print(name, "piqa", out["piqa"], flush=True)
    out["arc_easy"] = ES.arc_easy(model, enc, V, n=(150 if quick else 2376)); print(name, "arc_easy", out["arc_easy"], flush=True)
    out["humaneval"] = ES.humaneval(model, enc, V, itos, limit=(8 if quick else None), keep_ctx=keep_ctx); print(name, "humaneval", {k: v for k, v in out["humaneval"].items() if k != "samples"}, flush=True)
    out["code_syntax"] = ES.code_syntax_validity(model, enc, V, itos, D, n_prompts=(8 if quick else 120), keep_ctx=keep_ctx); print(name, "code_syntax", out["code_syntax"], flush=True)
    out["samples_T1"] = {n: ES.generate_greedy(model, enc, V, p, 350, itos, temperature=1.0, rng=np.random.default_rng(0), keep_ctx=keep_ctx) for n, p in SAMPLE_PROMPTS}

if __name__ == "__main__":
    tag = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000; quick = len(sys.argv) > 3 and sys.argv[3] == "quick"
    train, val, itos = harness.load(); V = len(itos); enc = ES.make_encoder(itos); D = os.path.join(ROOT, "data", "mix1")
    ch = {**default_chains(), **LEX}
    m = Ixchel(name=f"Ixchel-v0.4-mix-{tag}", verbose=True, chains=ch, mix="poe", cache_windows=(),
               xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),)); m.keep_dev = True
    t0 = time.time(); c0 = time.process_time(); m.fit(train, V)
    out = dict(tag=tag, n_train=int(len(train)), n_params=int(m.n_params()), fit_wall_s=time.time() - t0, fit_cpu_s=time.process_time() - c0,
               poe_w=dict(zip(m.expert_names(), [float(x) for x in m.poe_w])), dev_bpc_mix=m.dev_bpc_mix)
    try:
        import psutil; out["peak_rss_mb"] = psutil.Process().memory_info().peak_wset / 2**20
    except Exception: pass
    json.dump(out, open(os.path.join(HERE, "results", f"mixfinal_{tag}.json"), "w"), indent=1); print("fit", out["n_params"], out["fit_wall_s"], flush=True)
    ixo = dict(); battery("ixchel", m, enc, itos, V, D, quick, ixo); out["ixchel"] = ixo
    json.dump(out, open(os.path.join(HERE, "results", f"mixfinal_{tag}.json"), "w"), indent=1)
    # ---- ensamble con la GRU
    wfile = os.path.join(R3G, f"weights_gru_mix_{steps}.pt")
    while not os.path.exists(wfile): time.sleep(30)
    time.sleep(5); gru = GRUWrap(V, wfile)
    sel = np.arange(0, len(m.dev_pos_used), 3); dpos = m.dev_pos_used[sel]; xs = train[dpos]
    Pg = np.concatenate([gru.predict(train, dpos[s:s + 4096]) for s in range(0, len(dpos), 4096)]).astype(np.float64)
    LPg = np.log(np.maximum(Pg, 1e-12)).astype(np.float32); LPi = m.dev_logp[sel]
    w = poe_global(np.stack([LPi, LPg]), xs)
    out["ensemble_w"] = w.tolist(); out["dev_ixchel"] = float(-LPi[np.arange(len(xs)), xs].mean() / np.log(2)); out["dev_gru"] = float(-LPg[np.arange(len(xs)), xs].mean() / np.log(2))
    print("ensemble w", w, "dev ixchel/gru", out["dev_ixchel"], out["dev_gru"], flush=True)
    eo = dict(); battery("ensemble", Ensemble(m, gru, w), enc, itos, V, D, quick, eo, keep_ctx=1500); out["ensemble"] = eo
    json.dump(out, open(os.path.join(HERE, "results", f"mixfinal_{tag}.json"), "w"), indent=1); print("DONE", flush=True)

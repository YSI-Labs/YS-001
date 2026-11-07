"""GRU (0.84M, AdamW; baseline neuronal, NO es la arquitectura candidata) entrenada en mix1 por NUMERO DE PASOS fijo y evaluada con la
misma bateria que Ixchel (ixchel/eval_suite.py). Uso: python -I run_gru_mix.py <steps> [quick]"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "mix1"
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "6"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common")); sys.path.insert(0, os.path.join(ROOT, "ixchel"))
import numpy as np, torch, harness, eval_suite as ES
torch.set_num_threads(6)
from model import NeuralLM

class GRUScorer:
    """Envuelve NeuralLM: predict (ventana 128, contrato del harness) + score_sequence (pasada completa, para opcion multiple)."""
    def __init__(self, m): self.m = m; self.V = m.V
    def predict(self, ids, pos): return self.m.predict(ids, pos)
    @torch.no_grad()
    def score_sequence(self, ids, start):
        x = torch.from_numpy(np.asarray(ids[:-1], dtype=np.int64))[None, :]
        lp = torch.log_softmax(self.m.net(x)[0].float(), -1)           # [T-1, V]: pred del simbolo t+1 dado <= t
        tgt = torch.from_numpy(np.asarray(ids[1:], dtype=np.int64))
        ll = lp[torch.arange(len(tgt)), tgt][start - 1:].sum().item() / np.log(2)
        return ll

if __name__ == "__main__":
    steps = int(sys.argv[1]); quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
    train, val, itos = harness.load(); V = len(itos); enc = ES.make_encoder(itos)
    m = NeuralLM("gru", budget_s=1e9, max_steps=steps)
    t0 = time.time(); c0 = time.process_time(); m.fit(train, V); fit_wall, fit_cpu = time.time() - t0, time.process_time() - c0
    torch.save(m.net.state_dict(), os.path.join(HERE, f"weights_gru_mix_{steps}.pt"))
    g = GRUScorer(m); out = dict(arch="gru d256 L2 ctx128 AdamW", steps=steps, n_params=int(m.n_params()), fit_wall_s=fit_wall, fit_cpu_s=fit_cpu,
                                 tokens_seen=steps * 64 * 128)
    D = os.path.join(ROOT, "data", "mix1"); print("fit done", fit_wall, flush=True)
    dom = ES.domain_bpc(g, D, V, n_eval=3000 if quick else 30_000); nll = {d: dom[d].pop("nll") for d in dom}; out["domain_bpc"] = dom
    np.savez(os.path.join(HERE, f"gru_mix_{steps}_nll.npz"), **nll); print({d: round(v["bpc"], 3) for d, v in dom.items()}, flush=True)
    out["hellaswag"] = ES.hellaswag(g, enc, V, n=(200 if quick else 5000)); print("hellaswag", out["hellaswag"], flush=True)
    out["piqa"] = ES.piqa(g, enc, V, n=(150 if quick else 1838)); print("piqa", out["piqa"], flush=True)
    out["arc_easy"] = ES.arc_easy(g, enc, V, n=(150 if quick else 2376)); print("arc_easy", out["arc_easy"], flush=True)
    out["humaneval"] = ES.humaneval(g, enc, V, itos, limit=(8 if quick else None), keep_ctx=128); print("humaneval", {k: v for k, v in out["humaneval"].items() if k != "samples"}, flush=True)
    out["code_syntax"] = ES.code_syntax_validity(g, enc, V, itos, D, n_prompts=(8 if quick else 120), keep_ctx=128); print(out["code_syntax"], flush=True)
    samples = {}
    for name, prompt in [("web", "The history of the city begins"), ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
                         ("math", "Question: A farmer has 12 cows and buys 5 more. Each cow produces 3 liters of milk per day. How many liters in total per day?\nAnswer:"),
                         ("chat", "### Instruction:\nGive me three tips to learn to cook.\n\n### Response:\n"), ("books", "HAMLET.\nTo be, or not to be, ")]:
        samples[name] = ES.generate_greedy(g, enc, V, prompt, 350, itos, temperature=0.8, rng=np.random.default_rng(0), keep_ctx=128)
    out["samples"] = samples
    json.dump(out, open(os.path.join(HERE, f"gru_mix_{steps}.json"), "w"), indent=1); print("DONE", flush=True)

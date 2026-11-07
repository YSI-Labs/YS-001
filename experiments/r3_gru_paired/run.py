"""Corrida final reproducible del baseline neuronal.

Uso:  python -I run.py <transformer|gru> [budget_s=480] [n_eval=10000] [gen_chars=400]
Escribe result_<arch>.json en esta carpeta (via harness.run(out_json=...)).
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I no anade el directorio del script
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import harness  # noqa: E402
from model import NeuralLM  # noqa: E402

if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "transformer"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 480.0
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
    gen = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    out = os.path.join(HERE, f"result_{arch}.json")

    m = NeuralLM(arch, budget_s=1e9, max_steps=int(budget))
    res = harness.run(m, n_eval=n_eval, gen_chars=gen, seed=0, extra=m.info, out_json=out)
    harness.summarize(res)
    import numpy as np, torch
    torch.save(m.net.state_dict(), os.path.join(HERE, f"weights_{arch}.pt"))
    train, val, itos = harness.load()
    pos = harness.eval_positions(val, n_eval)
    nll = np.zeros(len(pos))
    for s in range(0, len(pos), 2048):
        pp = pos[s:s+2048]
        P = m.predict(val, pp).astype(np.float64)
        nll[s:s+len(pp)] = -np.log2(np.maximum(P[np.arange(len(pp)), val[pp]], 1e-12))
    np.save(os.path.join(HERE, f"nll_{arch}_n{n_eval}.npy"), nll)
    print("nll saved", nll.mean())
    print("train_s", res["train_s"], "steps", res["steps"], "tokens", res["tokens_seen"],
          "train_chars_per_s", round(res["train_chars_per_s"]), "eval_chars_per_s", round(res["eval_chars_per_s"]),
          "leak_free", res.get("leak_free"), "n_eval", n_eval)

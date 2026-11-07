"""Corrida final de un optimizador learning-rate-free sobre la GRU de referencia.

Uso:  python -I run_lrfree.py <dog|prodigy|coinkt> <seed> <tag> [opt_kw_json='{}'] [steps=2600] [n_eval=10000]
Escribe result_<tag>.json, nll_<tag>.npy (NLL por posicion, para test pareado vs el baseline) en esta carpeta.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -I no anade el directorio del script
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))

import json  # noqa: E402

import numpy as np  # noqa: E402

import harness  # noqa: E402
from model_lrfree import LRFreeGRU  # noqa: E402

if __name__ == "__main__":
    opt = sys.argv[1]
    seed = int(sys.argv[2])
    tag = sys.argv[3]
    kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2600
    n_eval = int(sys.argv[6]) if len(sys.argv) > 6 else 10_000

    m = LRFreeGRU(opt, opt_kw=kw, seed=seed, steps=steps)
    res = harness.run(m, n_eval=n_eval, gen_chars=400, seed=0, extra=m.info,
                      out_json=os.path.join(HERE, f"result_{tag}.json"))
    harness.summarize(res)

    train, val, itos = harness.load()
    pos = harness.eval_positions(val, n_eval)
    nll = np.zeros(len(pos))
    for s in range(0, len(pos), 2048):
        pp = pos[s:s + 2048]
        P = m.predict(val, pp).astype(np.float64)
        nll[s:s + len(pp)] = -np.log2(np.maximum(P[np.arange(len(pp)), val[pp]], 1e-12))
    np.save(os.path.join(HERE, f"nll_{tag}.npy"), nll)
    print("nll mean", nll.mean(), "| steps", res["steps"], "| train_cpu_s", m.info.get("train_cpu_s"),
          "| diverged_at", m.info.get("diverged_at_step"), "| leak_free", res.get("leak_free"))

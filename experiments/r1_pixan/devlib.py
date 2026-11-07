"""Seleccion SOLO con dev = ultimo 5% de train (val nunca se toca aqui)."""
import os, sys, time, json
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, HERE)
import harness

def dev_split():
    train, val, itos = harness.load()
    V = len(itos)
    cut = int(len(train) * 0.95)
    fit_part = train[:cut]
    dev_arr = train[cut - 1024:]  # 1024 chars de contexto previo (solo contexto, no objetivos)
    return fit_part, dev_arr, V

def dev_eval(model, n_eval=20000):
    fit_part, dev_arr, V = dev_split()
    t0 = time.time()
    model.fit(fit_part, V)
    fit_s = time.time() - t0
    res = harness.evaluate(model, dev_arr, V, n_eval=n_eval, check_leak=False)
    res.update(name=model.name, n_params=int(model.n_params()), fit_wall_s=fit_s,
               train_chars_per_s=len(fit_part) / max(fit_s, 1e-9), split="dev")
    return res

def log(rows, path):
    json.dump(rows, open(path, "w"), indent=1)
    for r in rows:
        print(f"{r['name']:<40} dev_bpc={r['bpc']:.4f} params={r['n_params']:>10,} fit={r['fit_wall_s']:6.1f}s "
              f"eval_ch/s={r['eval_chars_per_s']:,.0f}", flush=True)

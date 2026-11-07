"""Verifica que core_opt produce EXACTAMENTE las mismas probabilidades que core (mismo train, mismos pesos).
Uso: python -I test_exact.py <n_train_chars>"""
import os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import harness
import core, core_opt

n_tr = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
train, val, itos = harness.load()
V = len(itos)
res = {}
models = {}
for name, mod in (("core", core), ("core_opt", core_opt)):
    allc = mod.default_chains()
    ch = {k: allc[k] for k in "AWSL"}
    m = mod.Ixchel(chains=ch, gate_chains=("A", "W"), gated=True, name=name, verbose=False)
    t = time.time(); m.fit(train[:n_tr], V); res[name + "_fit_s"] = time.time() - t
    models[name] = m
pos = np.linspace(64, len(val) - 1, 4096).astype(np.int64)
P1 = models["core"].predict(val, pos)
P2 = models["core_opt"].predict(val, pos)
res["max_abs_diff_probs"] = float(np.abs(P1.astype(np.float64) - P2).max())
res["identical_bits"] = bool(np.array_equal(P1, P2))
# tablas: mismas claves y conteos
ok = True
for nm in models["core"].tabs:
    a, b = models["core"].tabs[nm], models["core_opt"].tabs[nm]
    for k in ("ckey", "cstart", "psym", "pcnt"):
        ok &= np.array_equal(a[k], b[k])
res["tables_identical"] = bool(ok)
# bpc sobre 20k posiciones (harness)
for name, m in models.items():
    r = harness.evaluate(m, val, V, n_eval=20000, check_leak=False)
    res[name + "_bpc_20k"] = r["bpc"]
    res[name + "_pred1_ms"] = None
# una posicion
t = time.time(); models["core"].predict(val[:2000], np.array([1000])); res["core_pred1_ms"] = (time.time() - t) * 1000
t = time.time(); models["core_opt"].predict(val[:2000], np.array([1000])); res["core_opt_pred1_ms"] = (time.time() - t) * 1000
print(res)

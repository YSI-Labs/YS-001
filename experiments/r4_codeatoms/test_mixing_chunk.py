"""Verifica que mixing.py (E por bloques) == ixchel/mixing.py (original) en pesos PoE."""
import sys, os, importlib.util, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
A = load(os.path.join(ROOT, "ixchel", "mixing.py"), "mix_orig"); B = load(os.path.join(HERE, "mixing.py"), "mix_chunk")
rng = np.random.default_rng(0)
K, m, V = 5, 50_000, 40
LP = np.log(rng.dirichlet(np.ones(V) * 0.3, size=(K, m)).astype(np.float32) + 1e-9).astype(np.float32)
xs = rng.integers(0, V, size=m).astype(np.int64)
wa = A.poe_global(LP, xs, iters=10); wb = B.poe_global(LP, xs, iters=10)
print("w original", np.round(wa, 6).tolist()); print("w chunk   ", np.round(wb, 6).tolist()); print("max abs diff", float(np.abs(wa - wb).max()))

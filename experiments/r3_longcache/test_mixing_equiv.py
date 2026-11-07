"""Compara poe_global nuevo (r3) con el original (ixchel/mixing.py) en datos aleatorios."""
import sys, importlib.util, numpy as np
HERE = __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
spec = importlib.util.spec_from_file_location("orig", HERE + "/ixchel/mixing.py"); orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import mixing as new
rng = np.random.default_rng(0)
K, m, V = 6, 3000, 27
logits = rng.normal(size=(K, m, V)) * 1.5 + rng.normal(size=(1, m, V))
LP = (logits - np.log(np.exp(logits).sum(2, keepdims=True))).astype(np.float32)
xs = rng.integers(0, V, size=m)
w1 = orig.poe_global(LP, xs); w2 = new.poe_global(LP, xs)
print("w orig", np.round(w1, 6).tolist()); print("w new ", np.round(w2, 6).tolist()); print("max abs diff", float(np.abs(w1 - w2).max()))

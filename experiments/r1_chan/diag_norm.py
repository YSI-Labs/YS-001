"""Diagnostico (dev2 solamente): el puntaje proto NO esta normalizado por la masa del contexto.
Compara softmax(beta*z) con p = relu(z)/sum relu(z) (mezclado con unigrama 1e-3) en dev2."""
import sys, json
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'experiments', 'r1_chan'))
import numpy as np, harness
from model import ChanHDC, _softmax
train, val, itos = harness.load(); V = len(itos)
cut = int(len(train) * 0.95); tr, dev = train[:cut], train[cut:]
half = len(dev) // 2
p2 = harness.eval_positions(dev[half:], 20_000) + half
y2 = dev[p2]
def nll(P): return float(-np.log2(np.maximum(P[np.arange(len(y2)), y2], 1e-12)).mean())
out = {}
for D, K in [(4096, 3), (4096, 8)]:
    m = ChanHDC(D=D, K=K, mode="proto").fit(tr, V)
    z = m.raw_logits(dev, p2).astype(np.float64)
    uni = (np.bincount(tr, minlength=V) + 1) / (len(tr) + V)
    r = np.maximum(z, 0) + 1e-12
    Pn = r / r.sum(1, keepdims=True)
    Pn = 0.999 * Pn + 0.001 * uni
    out[f"D{D}_K{K}_relu_norm_mix"] = nll(Pn)
    out[f"D{D}_K{K}_softmax_best_beta_grid"] = min(nll(_softmax(b * z)) for b in 10.0 ** np.linspace(-2, 8, 41))
    print(out, flush=True)
json.dump(out, open("diag_norm.json", "w"), indent=1)

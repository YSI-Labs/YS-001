"""Chequeo de cordura de los optimizadores en regresion lineal ruidosa (minibatch). No es parte del resultado."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
torch.set_num_threads(2)
from lrfree_opt import DoG, Prodigy, CoinKT

torch.manual_seed(0)
D, N = 50, 4000
X = torch.randn(N, D)
w_true = torch.randn(D)
y = X @ w_true + 0.5 * torch.randn(N)


def run(make, steps=1500, bs=64):
    torch.manual_seed(1)
    w = torch.zeros(D, requires_grad=True)
    w.data.normal_(0, 0.1)
    opt = make([w])
    for t in range(steps):
        idx = torch.randint(0, N, (bs,))
        loss = ((X[idx] @ w - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    full = float(((X @ w - y) ** 2).mean())
    return full, w


opt_ref = 0.25  # varianza del ruido: el optimo tiene MSE ~ 0.25
for name, mk in [
    ("Adam lr=1e-2 (referencia con LR)", lambda ps: torch.optim.Adam(ps, lr=1e-2)),
    ("DoG reps=1e-6", lambda ps: DoG(ps, reps=1e-6)),
    ("Prodigy d0=1e-6", lambda ps: Prodigy(ps, d0=1e-6)),
    ("CoinKT W0=1e-3", lambda ps: CoinKT(ps, wealth0=1e-3)),
    ("CoinKT W0=1e-1", lambda ps: CoinKT(ps, wealth0=1e-1)),
]:
    mse, w = run(mk)
    dist = float((w.detach() - w_true).norm())
    print(f"{name:34s} MSE_full={mse:.4f} (optimo~{opt_ref})  ||w-w*||={dist:.3f}  ||w||={float(w.norm()):.2f}")

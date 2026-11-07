"""Dev sweep 2 (solo dev; val intacto). Fit en train[:95%]. zthr (umbral de ruido) se elige en dev1, se reporta en dev2."""
import os, sys, time, json
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'experiments', 'r1_chan'))
import numpy as np, harness
from model import ChanHDC, _softmax
train, val, itos = harness.load(); V = len(itos)
cut = int(len(train) * 0.95)
tr, dev = train[:cut], train[cut:]
half = len(dev) // 2
p1 = harness.eval_positions(dev[:half], 20_000)
p2 = harness.eval_positions(dev[half:], 20_000) + half
def nll(P, y): return float(-np.log2(np.maximum(P[np.arange(len(y)), y], 1e-12)).mean())
rows = []
def go(D, K, H):
    m = ChanHDC(D=D, K=K, mode="wb", H=H, zthr=0.0)
    t0 = time.time(); m.fit(tr, V); fit = time.time() - t0
    grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    res = {}
    for z in grid:
        m.zthr = z
        res[z] = nll(m.predict(dev, p1).astype(np.float64), dev[p1])
    zb = min(grid, key=lambda z: res[z])
    m.zthr = zb
    t0 = time.time(); P2 = m.predict(dev, p2); inf = time.time() - t0
    r = dict(name=f"Chan-wb-D{D}-K{K}-H{H}", D=D, K=K, H=H, n_params=m.n_params(), fit_wall_s=round(fit, 1),
             zthr_dev1=zb, dev1_bpc_by_z={str(k): round(v, 4) for k, v in res.items()},
             dev2_bpc=nll(P2.astype(np.float64), dev[p2]), infer_chars_per_s=round(len(p2) / inf))
    print(json.dumps(r), flush=True); rows.append(r)
    del m
for D, K, H in [(1024, 5, 1), (2048, 5, 1), (2048, 5, 31), (2048, 5, 127), (4096, 5, 31), (2048, 3, 127), (2048, 8, 127)]:
    go(D, K, H)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_dev_wb2.json"), "w"), indent=1)

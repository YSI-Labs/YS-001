"""Corrida final reproducible de Chan. Uso: python -I run.py <variante>
Hiperparametros de escala/umbral se eligen SOLO en dev (ultimo 5% del train; fit en train[:95%], dev1 para elegir,
dev2 para reportar en el sweep). Luego se reentrena con TODO el train y se evalua UNA vez en val (harness.run)."""
import os, sys, json, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, harness
from model import ChanHDC, _softmax

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {
    # HDC puro: un prototipo por caracter, superposicion de ordenes 1..8, beta de softmax tuneado en dev
    "proto": dict(make=lambda: ChanHDC(D=4096, K=8, mode="proto"), tune="beta"),
    # Fix: memoria WB por orden con conteos HDC, particionada por hash de (x[t-1],x[t-2]) mod H, umbral de ruido z*sigma
    # zthr=4.0 y H=31 elegidos en sweep2 (dev1). Sin re-barrido aqui por coste de inferencia (~220 ch/s con H=31).
    "wb": dict(make=lambda: ChanHDC(D=2048, K=5, mode="wb", H=31), tune="fixed", zthr=4.0),
}

def nll(P, y):
    return float(-np.log2(np.maximum(P[np.arange(len(y)), y], 1e-12)).mean())

def tune(name, cfg):
    if cfg["tune"] == "fixed":
        return dict(zthr=cfg["zthr"], note="fixed from sweep_dev_wb2 (dev1 choice); not re-tuned in run.py")
    train, val, itos = harness.load(); V = len(itos)
    cut = int(len(train) * 0.95); tr, dev = train[:cut], train[cut:]
    half = len(dev) // 2
    p1 = harness.eval_positions(dev[:half], 20_000)
    y1 = dev[p1]
    m = cfg["make"]().fit(tr, V)
    if cfg["tune"] == "beta":
        z1 = m.raw_logits(dev, p1).astype(np.float64)
        grid = 10.0 ** np.linspace(-2, 8, 51)
        b = min(grid, key=lambda b: nll(_softmax(b * z1), y1))
        b = min(b * 10.0 ** np.linspace(-0.2, 0.2, 21), key=lambda b: nll(_softmax(b * z1), y1))
        return dict(beta=float(b))
    grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    sc = {}
    for z in grid:
        m.zthr = z
        sc[z] = nll(m.predict(dev, p1).astype(np.float64), y1)
    return dict(zthr=min(grid, key=lambda z: sc[z]), dev1_bpc_by_z={str(k): v for k, v in sc.items()})

if __name__ == "__main__":
    v = sys.argv[1]
    cfg = VARIANTS[v]
    t0 = time.time()
    chosen = tune(v, cfg)
    print("tuned on dev1:", chosen, f"({time.time()-t0:.0f}s)", flush=True)
    m = cfg["make"]()
    if "beta" in chosen: m.beta = chosen["beta"]
    if "zthr" in chosen: m.zthr = chosen["zthr"]
    res = harness.run(m, n_eval=50_000, gen_chars=400, out_json=os.path.join(HERE, f"result_{v}.json"),
                      extra=dict(tuning=chosen, protocol="tuned on dev1 (last 5% of train, first half); final fit on full train; one val eval"))
    harness.summarize(res)

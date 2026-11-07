"""Barrido en DEV reservado del train (ultimo 5% de train). val NO se toca aqui.
Protocolo: fit en train[:95%]; beta (proto) se elige en dev1 (primera mitad del dev) y se reporta en dev2."""
import os, sys, time, json
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'experiments', 'r1_chan'))
import numpy as np, harness
from model import ChanHDC, _softmax
from baseline_wb import WittenBell

train, val, itos = harness.load(); V = len(itos)
cut = int(len(train) * 0.95)
tr, dev = train[:cut], train[cut:]
half = len(dev) // 2
dev1, dev2 = dev[:half], dev[half:]
N_EVAL = 20_000
pos1 = harness.eval_positions(dev1, N_EVAL)
pos2 = harness.eval_positions(dev2, N_EVAL)
# dev1 y dev2 se concatenan en un solo array para el predict (posiciones absolutas dentro de 'dev')
devall = dev
p1 = pos1
p2 = pos2 + half

def nll_bits(P, y):
    return float(-np.log2(np.maximum(P[np.arange(len(y)), y], 1e-12)).mean())

def run_cfg(model):
    t0 = time.time(); model.fit(tr, V); fit = time.time() - t0
    out = dict(name=model.name, n_params=int(model.n_params()), fit_wall_s=round(fit, 2),
               train_chars_per_s=round(len(tr) / fit))
    if model.mode == "proto":
        z1 = model.raw_logits(devall, p1).astype(np.float64)
        y1 = devall[p1]
        grid = 10.0 ** np.linspace(-2, 8, 51)
        best = min(grid, key=lambda b: nll_bits(_softmax(b * z1), y1))
        # refina alrededor del mejor (busqueda 1-D)
        grid2 = best * 10.0 ** np.linspace(-0.2, 0.2, 21)
        best = min(grid2, key=lambda b: nll_bits(_softmax(b * z1), y1))
        model.beta = float(best)
        out["beta_dev1"] = float(best)
    t0 = time.time()
    P2 = model.predict(devall, p2)
    out["dev2_bpc"] = nll_bits(P2.astype(np.float64), devall[p2])
    out["infer_chars_per_s"] = round(len(p2) / (time.time() - t0))
    out["dev2_bpc_wb_refs"] = None
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = []
    if which in ("ref", "all"):
        wb = WittenBell(5); wb.fit(tr, V)
        P2 = wb.predict(devall, p2)
        r = dict(name="WB-ngram-5(ref)", n_params=wb.n_params(), dev2_bpc=nll_bits(P2.astype(np.float64), devall[p2]))
        print(json.dumps(r), flush=True); cfgs.append(r)
    if which in ("proto", "all"):
        for K in (1, 2, 3, 4, 6, 8):
            cfgs.append(run_cfg(ChanHDC(D=4096, K=K, mode="proto")))
        for D in (1024, 16384):
            cfgs.append(run_cfg(ChanHDC(D=D, K=8, mode="proto")))
        for k in (3, 5):
            cfgs.append(run_cfg(ChanHDC(D=4096, K=k, mode="proto", orders=[k], name=f"Chan-proto-D4096-only-order{k}")))
    if which in ("wb", "all"):
        for D in (1024, 4096, 16384):
            cfgs.append(run_cfg(ChanHDC(D=D, K=5, mode="wb")))
        for K in (3, 8):
            cfgs.append(run_cfg(ChanHDC(D=4096, K=K, mode="wb")))
    json.dump(cfgs, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sweep_dev_{which}.json"), "w"), indent=1)

"""Variantes de calibracion (solo lectura de resid_d64.pt y base_full.npz): a escalar sin b, y a+b con b=0 en simbolos ausentes del dev.
Mismos datos que resid.py eval. Sin ajustar nada con val."""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import resid as R
from resid import Resid, f_logits, logsoft, fit_ab, harness, LN2

def fit_a_only(LP, Fz, x):
    """Newton 1-D sobre a (b=0): convexo."""
    ar = np.arange(len(x)); a = 1.0
    for _ in range(50):
        Q = np.exp(logsoft(a * LP + Fz)); m = (Q * LP).sum(1)
        g = (m - LP[ar, x]).sum(); H = ((Q * LP * LP).sum(1) - m * m).sum()
        a_new = a - g / H
        if abs(a_new - a) < 1e-10: a = a_new; break
        a = a_new
    return a

if __name__ == "__main__":
    train, val, itos = harness.load(); V = len(itos)
    net = Resid(V, 64); net.load_state_dict(torch.load(os.path.join(HERE, "resid_d64.pt"))); net.eval()
    bz = np.load(os.path.join(HERE, "base_full.npz"))
    dev_pos, dev_lp = bz["dev_pos"], bz["dev_logp"].astype(np.float64)
    pos_v, LPv = bz["pos_v"], bz["LPv"].astype(np.float64)
    xd, xv = train[dev_pos], val[pos_v]
    Fd = f_logits(net, train, dev_pos, V).astype(np.float64)
    Fv = f_logits(net, val, pos_v, V).astype(np.float64)
    a1 = fit_a_only(dev_lp, Fd, xd)
    ar = np.arange(len(xv))
    res = {}
    res["dev_a_only_fit"] = a1
    res["dev_bpc_a_only"] = float(-logsoft(a1 * dev_lp + Fd)[np.arange(len(xd)), xd].mean() / LN2)
    res["val_bpc_a_only"] = float(-logsoft(a1 * LPv + Fv)[ar, xv].mean() / LN2)
    # simbolos con pocos casos en dev: cuantos del val caen en simbolos con <20 casos en dev
    cnt_d = np.bincount(xd, minlength=V); cnt_v = np.bincount(xv, minlength=V)
    rare = cnt_d < 20
    res["dev_symbols_lt20"] = int(rare.sum()); res["val_positions_on_those_symbols"] = int(rare[xv].sum())
    th, _, _ = fit_ab(dev_lp, Fd, xd)
    nll_ab = -logsoft(th[0] * LPv + th[1:][None, :] + Fv)[ar, xv] / LN2
    res["val_bpc_ab_fit"] = float(nll_ab.mean())
    res["val_ab_fit_on_rare_symbols_bpc_mean"] = float(nll_ab[rare[xv]].mean()) if rare[xv].any() else None
    res["b_on_rare_symbols_mean"] = float(th[1:][rare].mean()) if rare.any() else None
    res["b_on_common_symbols_mean"] = float(th[1:][~rare].mean()) if (~rare).any() else None
    json.dump(res, open(os.path.join(HERE, "result_variants_d64.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

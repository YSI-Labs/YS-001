"""SE pareados (bloques de 500 y iid) para la variante a-escalar frente a Ixchel base y frente al ensamble (mismos pesos PoE)."""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from resid import Resid, f_logits, logsoft, harness, LN2, W_ENS, block_stats, GRU_W
from eval_variants import fit_a_only
from gru_model import NeuralLM

if __name__ == "__main__":
    train, val, itos = harness.load(); V = len(itos)
    net = Resid(V, 64); net.load_state_dict(torch.load(os.path.join(HERE, "resid_d64.pt"))); net.eval()
    bz = np.load(os.path.join(HERE, "base_full.npz"))
    dev_pos, dev_lp = bz["dev_pos"], bz["dev_logp"].astype(np.float64)
    pos_v, LPv = bz["pos_v"], bz["LPv"].astype(np.float64)
    xd, xv = train[dev_pos], val[pos_v]; ar = np.arange(len(xv))
    Fd = f_logits(net, train, dev_pos, V).astype(np.float64)
    Fv = f_logits(net, val, pos_v, V).astype(np.float64)
    a1 = fit_a_only(dev_lp, Fd, xd)
    nll_base = -LPv[ar, xv] / LN2
    nll_a = -logsoft(a1 * LPv + Fv)[ar, xv] / LN2
    g = NeuralLM("gru", budget_s=1e9, max_steps=1); g.V = V
    g.net = g._build(V); g.net.load_state_dict(torch.load(GRU_W)); g.net.eval(); torch.set_num_threads(2)
    Pg = g.predict(val, pos_v).astype(np.float64)
    nll_ens = -logsoft(W_ENS[0] * LPv + W_ENS[1] * np.log(np.maximum(Pg, 1e-12)))[ar, xv] / LN2
    out = dict(a_only=a1, val_a_only_bpc=float(nll_a.mean()), val_ens_bpc=float(nll_ens.mean()), val_base_bpc=float(nll_base.mean()),
               paired_base_minus_a_only_block=block_stats(nll_base - nll_a),
               paired_ens_minus_a_only_block=block_stats(nll_ens - nll_a),
               paired_ens_minus_a_only_iid_se=float((nll_ens - nll_a).std(ddof=1) / np.sqrt(len(ar))))
    json.dump(out, open(os.path.join(HERE, "result_paired_a_only_d64.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))

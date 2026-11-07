"""Test pareado y ensamble Ixchel + GRU (Shakespeare). La GRU (r3_gru_paired) NO se entreno con el ultimo 5% del train,
igual que el dev interno de Ixchel -> los pesos del ensamble se ajustan en dev sin fuga; val solo para el reporte."""
import os, sys, json, time, importlib.util
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, torch, harness, run_v02
from core_v02 import Ixchel
from mixing import poe_global
torch.set_num_threads(4)
R3 = os.path.join(ROOT, "experiments", "r3_gru_paired")
spec = importlib.util.spec_from_file_location("gru_model", os.path.join(R3, "model.py")); gm = importlib.util.module_from_spec(spec); spec.loader.exec_module(gm)

def load_gru(V):
    g = gm.NeuralLM("gru", budget_s=1e9, max_steps=1); g.V = V
    g.net = g._build(V); g.net.load_state_dict(torch.load(os.path.join(R3, "weights_gru.pt"))); g.net.eval(); return g

def block_stats(x, block=500):
    nb = len(x) // block; bm = x[:nb*block].reshape(nb, block).mean(1)
    return float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x))), float(bm.std(ddof=1)/np.sqrt(nb))

def nll_of(P, val, pos): return -np.log2(np.maximum(P[np.arange(len(pos)), val[pos]].astype(np.float64), 1e-12))

if __name__ == "__main__":
    train, val, itos = harness.load(); V = len(itos)
    gru = load_gru(V)
    pos = harness.eval_positions(val, 10_000)
    nll_gru = np.load(os.path.join(R3, "nll_gru_n10000.npy"))
    chk = nll_of(gru.predict(val, pos[:500]), val, pos[:500]); assert abs(chk.mean() - nll_gru[:500].mean()) < 1e-3, "GRU recarga no coincide"
    out = {"gru_bpc_10k": float(nll_gru.mean())}
    for cfg in sys.argv[1:]:
        m = Ixchel(name=f"Ixchel-{cfg}", verbose=False, **run_v02.CFG[cfg]); m.keep_dev = True
        m.fit(train, V)
        # Ixchel en val (10k)
        P_i = np.concatenate([m.predict(val, pos[s:s+4096]) for s in range(0, len(pos), 4096)]).astype(np.float64)
        nll_i = nll_of(P_i, val, pos)
        # dev: ixchel (tablas de train[:cut]) vs GRU en 50k posiciones de dev
        sel = np.arange(0, len(m.dev_pos_used), 3); dpos = m.dev_pos_used[sel]
        LPi = m.dev_logp[sel]
        Pg = np.concatenate([gru.predict(train, dpos[s:s+4096]) for s in range(0, len(dpos), 4096)]).astype(np.float64)
        LPg = np.log(np.maximum(Pg, 1e-12)).astype(np.float32)
        xs = train[dpos]
        dev_i = float(-(LPi[np.arange(len(dpos)), xs]).mean()/np.log(2)); dev_g = float(-(LPg[np.arange(len(dpos)), xs]).mean()/np.log(2))
        w = poe_global(np.stack([LPi, LPg]), xs)
        # ensamble en val
        Pg_v = np.concatenate([gru.predict(val, pos[s:s+4096]) for s in range(0, len(pos), 4096)]).astype(np.float64)
        L = w[0]*np.log(np.maximum(P_i, 1e-12)) + w[1]*np.log(np.maximum(Pg_v, 1e-12))
        L -= L.max(1, keepdims=True); Pe = np.exp(L); Pe /= Pe.sum(1, keepdims=True)
        nll_e = nll_of(Pe, val, pos)
        d_ig = nll_gru - nll_i   # >0 => Ixchel mejor
        d_eg = nll_gru - nll_e
        d_ei = nll_i - nll_e
        r = dict(ixchel_bpc=block_stats(nll_i), gru_bpc=block_stats(nll_gru), ens_bpc=block_stats(nll_e), w_ens=w.tolist(),
                 dev_ixchel=dev_i, dev_gru=dev_g,
                 paired_gru_minus_ixchel=block_stats(d_ig), paired_gru_minus_ens=block_stats(d_eg), paired_ixchel_minus_ens=block_stats(d_ei),
                 frac_positions_ixchel_better=float((d_ig > 0).mean()))
        out[cfg] = r
        print(cfg, json.dumps(r), flush=True)
        json.dump(out, open(os.path.join(HERE, "results", "ensemble_test.json"), "w"), indent=1)

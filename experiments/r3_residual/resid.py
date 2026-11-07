"""Residuo neuronal f_theta sobre log p_ixchel(x|ctx) (modelo log-lineal):
      p(x | ctx) = softmax_x( a * LP[x] + b[x] + f_theta(ctx)[x] ),   LP = log p_ixchel v0.3 (base fija).
  - train: f_theta (GRU 1 capa d configurable + cabeza lineal) entrenada con a=1, b=0 sobre LP OUT-OF-FOLD
           (4 bloques contiguos de train[:cut], ver base_fit.py). Optimizador: AdamW lr 3e-3 coseno
           (PRIMERA MEDICION, etiquetada como AdamW; el brief lo permite en esta ronda). Pasos FIJOS.
  - eval:  a (escalar) y b (vector V) por Newton (problema CONVEXO en (a,b)) SOLO con el dev interno
           (ultimo 5% de train, LP de tablas train[:cut]). Val solo para el reporte.
           Comparacion: Ixchel v0.3 solo (val 10k) y ensamble Ixchel+GRU (pesos de ixchel/results/ensemble_test.json).
Uso: python -I resid.py train --d 128 --steps 2600 --tag d128
     python -I resid.py eval  --d 128 --tag d128
"""
import os, sys, time, json, math, argparse
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
import harness
from base_fit import M_SEG, K, CTX_MIN, blocks

T = 128          # contexto / longitud de ventana (igual que la GRU baseline)
LN2 = math.log(2)
W_ENS = [0.7082219786371106, 0.33000295046453587]   # pesos PoE Ixchel/GRU de ixchel/results/ensemble_test.json (v03)
GRU_W = __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'experiments', 'r3_gru_paired', 'weights_gru.pt')


class Resid(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.head(h)


def load_oof(train, V):
    cut, B = blocks(train)
    seg = np.zeros((K, M_SEG), np.uint8)
    LP = np.zeros((K, M_SEG, V), np.float16)
    for k in range(K):
        seg[k] = train[k * B: k * B + M_SEG]
        LP[k, CTX_MIN:] = np.load(os.path.join(HERE, f"lp_fold{k}.npy"))
    return seg, LP


def train_run(a):
    train, val, itos = harness.load(); V = len(itos)
    seg, LP = load_oof(train, V)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    net = Resid(V, a.d)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    ar = np.arange(T)
    cpu0, w0 = time.process_time(), time.time()
    run, curve = 0.0, []
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        f = rng.integers(0, K, a.batch)
        s = rng.integers(CTX_MIN, M_SEG - T, a.batch)
        inp = seg[f[:, None], s[:, None] - 1 + ar[None, :]]          # paso j: entrada = char t-1 (t = s+j)
        tix = s[:, None] + ar[None, :]
        tgt = seg[f[:, None], tix]
        lp = torch.from_numpy(LP[f[:, None], tix].astype(np.float32))  # [B,T,V]
        logits = net(torch.from_numpy(inp.astype(np.int64))) + lp     # a = 1, b = 0 durante el entrenamiento
        loss = F.cross_entropy(logits.reshape(-1, V), torch.from_numpy(tgt.astype(np.int64)).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        run += float(loss.item())
        if (step + 1) % 200 == 0:
            curve.append(dict(step=step + 1, lr=lr, train_bpc_resid_only=run / 200 / LN2,
                              cpu_s=time.process_time() - cpu0, wall_s=time.time() - w0))
            print(curve[-1], flush=True)
            run = 0.0
    cpu, wall = time.process_time() - cpu0, time.time() - w0
    torch.save(net.state_dict(), os.path.join(HERE, f"resid_{a.tag}.pt"))
    json.dump(dict(tag=a.tag, d=a.d, n_params_f=n_params, steps=a.steps, batch=a.batch, ctx=T, lr=a.lr,
                   schedule="warmup %d + cosine a 0, pasos fijos" % a.warmup, wd=0.1, betas=[0.9, 0.95], clip=1.0,
                   seed=a.seed, train_cpu_s=cpu, train_wall_s=wall, train_positions_oof=int(K * (M_SEG - CTX_MIN)),
                   curve=curve), open(os.path.join(HERE, f"train_{a.tag}.json"), "w"), indent=1)
    print("train done cpu", cpu, "wall", wall, "params", n_params, flush=True)


def f_logits(net, ids, pos, V, bs=4096):
    """f_theta(ctx)[.] para cada pos, con contexto = ids[pos-T:pos] (causal; nunca mira ids[pos:])."""
    out = np.empty((len(pos), V), np.float32)
    ar = np.arange(T)
    net.eval()
    with torch.no_grad():
        for s in range(0, len(pos), bs):
            pp = pos[s:s + bs]
            L = np.minimum(pp, T)
            idx = pp[:, None] - L[:, None] + ar[None, :]
            X = np.where(ar[None, :] < L[:, None], ids[np.clip(idx, 0, len(ids) - 1)], 0).astype(np.int64)
            lg = net(torch.from_numpy(X))
            out[s:s + len(pp)] = lg[torch.arange(len(pp)), torch.from_numpy(L - 1)].numpy()
    return out


def logsoft(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fit_ab(LP, Fz, x, iters=30):
    """Newton amortiguado para theta = (a, b[V]) minimizando NLL de softmax(a*LP + b + Fz). Convexo en theta."""
    N, V = LP.shape
    ar = np.arange(N)
    cnt = np.bincount(x, minlength=V).astype(np.float64)
    th = np.zeros(1 + V); th[0] = 1.0

    def probs(th):
        lq = logsoft(th[0] * LP + th[1:][None, :] + Fz)
        return lq, np.exp(lq)

    def nll(th):
        lq, _ = probs(th)
        return float(-lq[ar, x].sum())

    f0 = nll(th)
    hist = [f0]
    for it in range(iters):
        lq, Q = probs(th)
        m = (Q * LP).sum(1)
        g = np.empty(1 + V)
        g[0] = (m - LP[ar, x]).sum()
        g[1:] = Q.sum(0) - cnt
        H = np.empty((1 + V, 1 + V))
        H[0, 0] = ((Q * LP * LP).sum(1) - m * m).sum()
        H[0, 1:] = H[1:, 0] = (Q * (LP - m[:, None])).sum(0)
        H[1:, 1:] = np.diag(Q.sum(0)) - Q.T @ Q
        D = -np.linalg.solve(H + 1e-6 * np.eye(1 + V), g)      # ridge: direccion gauge de b (sin efecto)
        slope = float(g @ D)
        alpha = 1.0
        while True:
            f1 = nll(th + alpha * D)
            if f1 <= f0 + 1e-4 * alpha * slope or alpha < 1e-8:
                break
            alpha *= 0.5
        th = th + alpha * D
        hist.append(f1)
        done = abs(f0 - f1) < 1e-10 * N
        f0 = f1
        if done:
            break
    return th, f0, hist


def block_stats(x, block=500):
    nb = len(x) // block
    bm = x[:nb * block].reshape(nb, block).mean(1)
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))), float(bm.std(ddof=1) / np.sqrt(nb))]


def eval_run(a):
    train, val, itos = harness.load(); V = len(itos)
    net = Resid(V, a.d)
    net.load_state_dict(torch.load(os.path.join(HERE, f"resid_{a.tag}.pt"))); net.eval()
    bz = np.load(os.path.join(HERE, "base_full.npz"))
    dev_pos, dev_lp = bz["dev_pos"], bz["dev_logp"].astype(np.float64)
    pos_v, LPv = bz["pos_v"], bz["LPv"].astype(np.float64)
    xd, xv = train[dev_pos], val[pos_v]
    out = dict(tag=a.tag, d=a.d, n_params_f=sum(p.numel() for p in net.parameters()))
    # ---- DEV: ajuste de (a,b) por Newton (dev no se uso para entrenar f)
    Fd = f_logits(net, train, dev_pos, V).astype(np.float64)
    th, nll_d, hist = fit_ab(dev_lp, Fd, xd)
    dev_bpc = lambda Z, x: float(-Z[np.arange(len(x)), x].mean() / LN2)
    out["dev"] = dict(base_bpc=dev_bpc(dev_lp, xd),
                      resid_a1b0_bpc=dev_bpc(logsoft(dev_lp + Fd), xd),
                      resid_fit_bpc=float(nll_d / len(xd) / LN2), a=float(th[0]),
                      b_range=[float(th[1:].min()), float(th[1:].max())], newton_iters=len(hist) - 1,
                      n_dev=int(len(xd)))
    print("dev", out["dev"], flush=True)
    # ---- VAL (solo reporte): base, residuo con (a,b) de dev, residuo con a=1,b=0
    Fv = f_logits(net, val, pos_v, V).astype(np.float64)
    nll_base = -LPv[np.arange(len(xv)), xv] / LN2
    nll_res = -logsoft(th[0] * LPv + th[1:][None, :] + Fv)[np.arange(len(xv)), xv] / LN2
    nll_res10 = -logsoft(LPv + Fv)[np.arange(len(xv)), xv] / LN2
    out["val"] = dict(base_bpc=float(nll_base.mean()), resid_bpc=float(nll_res.mean()),
                      resid_a1b0_bpc=float(nll_res10.mean()), n_eval=int(len(xv)))
    out["val_paired_base_minus_resid"] = block_stats(nll_base - nll_res)      # >0 => residuo mejor
    out["frac_resid_better_than_base"] = float(((nll_base - nll_res) > 0).mean())
    # ---- ensamble Ixchel + GRU (pesos PoE de ixchel/results/ensemble_test.json, cfg v03)
    from gru_model import NeuralLM
    g = NeuralLM("gru", budget_s=1e9, max_steps=1); g.V = V
    g.net = g._build(V); g.net.load_state_dict(torch.load(GRU_W)); g.net.eval()
    torch.set_num_threads(2)
    Pg = g.predict(val, pos_v).astype(np.float64)
    nll_ens = -logsoft(W_ENS[0] * LPv + W_ENS[1] * np.log(np.maximum(Pg, 1e-12)))[np.arange(len(xv)), xv] / LN2
    nll_gru = -np.log2(np.maximum(Pg[np.arange(len(xv)), xv], 1e-12))
    out["val"].update(gru_bpc=float(nll_gru.mean()), ens_bpc=float(nll_ens.mean()))
    out["val_paired_ens_minus_resid"] = block_stats(nll_ens - nll_res)       # >0 => residuo mejor que el ensamble
    out["val_paired_base_minus_ens"] = block_stats(nll_base - nll_ens)
    out["val_paired_resid_a1b0_minus_fit"] = block_stats(nll_res10 - nll_res)
    # ---- chequeo de fuga: OOF (train, usado para entrenar f) vs dev (no usado para entrenar f)
    oof = [json.load(open(os.path.join(HERE, f"base_fold{k}.json")))["oof_bpc"] for k in range(K)]
    full = json.load(open(os.path.join(HERE, "base_full.json")))
    out["leak_check"] = dict(oof_base_bpc_per_fold=oof, oof_base_bpc_mean=float(np.mean(oof)),
                             dev_base_bpc=out["dev"]["base_bpc"], train_leaky_base_bpc_20k=full["bpc_train_leaky_20k"])
    json.dump(out, open(os.path.join(HERE, f"result_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="d128")
    a = ap.parse_args()
    train_run(a) if a.mode == "train" else eval_run(a)

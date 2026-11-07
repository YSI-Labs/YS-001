"""Entrena una GRU (2 capas, d=256, 0.84M params; AdamW lr 3e-3 coseno = CONTROL estandar, no es tecnologia propia) con los datos de un brazo del experimento de filtrado y
mide bpc sobre 400 docs limpios de C4 no vistos. Mismo numero de pasos/lote en todos los brazos (presupuesto fijo de computo).
Brazos N_* (pool limpio de 8000 docs): N_all, N_rand50 (50% al azar), N_low50 (50% con menor bpc KN), N_high50 (50% con mayor bpc KN).
Brazos C_* (pool con 30% de docs degradados): C_unf (todo), C_filtA (quita el 30% peor por bpc KN), C_filtFull (quita el 30% peor por bpc de Ixchel con cache), C_oracle (quita los degradados reales).
Uso: python -I filter_train.py <brazo> <semilla> [hilos=2] [pasos=1200]"""
import os, sys, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
BATCH, CTX = 32, 128

def select(arm, P):
    pool, junk, jidx, sc, scj = P["pool"], P["junk"], P["junk_idx"], P["sc_pool"], P["sc_junk"]; n = len(pool); allidx = np.arange(n)
    if arm.startswith("N_"):
        S = sc; docs = pool
        if arm == "N_all": keep = allidx
        elif arm == "N_rand50": keep = np.sort(np.random.default_rng(12345).choice(n, n // 2, replace=False))
        elif arm == "N_low50": keep = np.sort(np.argsort(S[:, 1])[: n // 2])
        elif arm == "N_high50": keep = np.sort(np.argsort(-S[:, 1])[: n // 2])
        else: raise SystemExit("brazo desconocido")
        return docs[keep]
    docs = pool.copy(); S = sc.copy(); docs[jidx] = junk; S[jidx] = scj; isj = np.zeros(n, bool); isj[jidx] = True; kdrop = int(0.30 * n)
    if arm == "C_unf": keep = allidx
    elif arm == "C_filtA": keep = np.sort(np.argsort(S[:, 1])[: n - kdrop])
    elif arm == "C_filtFull": keep = np.sort(np.argsort(S[:, 0])[: n - kdrop])
    elif arm == "C_oracle": keep = allidx[~isj]
    else: raise SystemExit("brazo desconocido")
    sel = np.zeros(n, bool); sel[keep] = True
    print("junk restante en el brazo:", int((isj & sel).sum()), "de", int(sel.sum()), "docs", flush=True)
    return docs[keep]

class GRULM(nn.Module):
    def __init__(self, V, d=256):
        super().__init__(); self.emb = nn.Embedding(V, d); self.rnn = nn.GRU(d, d, num_layers=2, batch_first=True); self.head = nn.Linear(d, V)
    def forward(self, x): h, _ = self.rnn(self.emb(x)); return self.head(h)

if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2]); threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2; steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    torch.set_num_threads(threads); torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
    P = np.load(os.path.join(ROOT, "data", "filter_exp", "pool.npz")); V = int(P["V"]); docs = select(arm, P); stream = torch.from_numpy(docs.reshape(-1).astype(np.int64)); ev = torch.from_numpy(P["eval"].astype(np.int64))
    net = GRULM(V); npar = sum(p.numel() for p in net.parameters()); opt = torch.optim.AdamW(net.parameters(), lr=3e-3); t0 = time.time(); N = len(stream)
    for s in range(steps):
        lr = 3e-3 * min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / steps))
        for g in opt.param_groups: g["lr"] = lr
        st = rng.integers(0, N - CTX - 1, BATCH); x = torch.stack([stream[a:a + CTX + 1] for a in st])
        loss = F.cross_entropy(net(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if s % 200 == 0: print(arm, seed, s, round(float(loss) / math.log(2), 3), f"{time.time() - t0:.0f}s", flush=True)
    net.eval()
    with torch.no_grad():
        lp = F.log_softmax(net(ev[:, :-1]), -1); nll = -lp.gather(-1, ev[:, 1:, None])[..., 0] / math.log(2)
    per_doc = nll[:, 63:].mean(1).numpy(); out = dict(arm=arm, seed=seed, steps=steps, batch=BATCH, ctx=CTX, params=npar, train_chars=int(len(stream)), eval_bpc=float(per_doc.mean()), train_s=time.time() - t0)
    os.makedirs(os.path.join(HERE, "results", "filter"), exist_ok=True)
    np.save(os.path.join(HERE, "results", "filter", f"{arm}_s{seed}_nll.npy"), per_doc); json.dump(out, open(os.path.join(HERE, "results", "filter", f"{arm}_s{seed}.json"), "w"))
    print("FILTER_TRAIN_DONE", arm, seed, round(out["eval_bpc"], 4), f"{out['train_s']:.0f}s", flush=True)

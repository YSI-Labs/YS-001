"""Baseline neuronal ESTANDAR de referencia (no es un candidato del proyecto).

Dos arquitecturas con backprop + AdamW + cosine, envueltas en el contrato del harness:
  - arch="transformer": decoder causal pequeno (pre-LN, GPT-style), d=128, 4 capas, 4 cabezas, ctx 128.
  - arch="gru":         GRU de 2 capas, d=256, entrenamiento truncado a 128 caracteres.

Presupuesto: TIEMPO de pared de entrenamiento (budget_s, por defecto 480 s = 8 min). El tiempo de las
evaluaciones de dev intermedias NO cuenta para el presupuesto (se registra aparte).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(6)


# ---------------------------------------------------------------- transformer (GPT-style, pre-LN)
class _Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        sh = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.fc2(F.gelu(self.fc(self.ln2(x))))


class _Transformer(nn.Module):
    def __init__(self, V, block, d, n_layer, n_head):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([_Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight  # weight tying (estandar nanoGPT)
        # init estandar nanoGPT: N(0, 0.02) en Linear/Embedding, bias 0. (Sin esto, el Embedding N(0,1)
        # atado a la cabeza da logits de std ~11 y el bpc inicial es absurdo.)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))


# ---------------------------------------------------------------- GRU
class _GRU(nn.Module):
    def __init__(self, V, d, n_layer):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.gru = nn.GRU(d, d, num_layers=n_layer, batch_first=True)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.head(h)


ARCHS = {
    "transformer": dict(d=128, n_layer=4, n_head=4, block=128),
    "gru": dict(d=256, n_layer=2, block=128),
}


class NeuralLM:
    """Contrato harness: fit(train, V), predict(ids, pos) -> [len(pos), V] normalizado, n_params()."""

    def __init__(self, arch="transformer", budget_s=480.0, batch=64, lr=3e-3, wd=0.1, betas=(0.9, 0.95),
                 warmup=20, clip=1.0, dev_frac=0.05, dev_positions=2000, checkpoints=(),
                 seed=0, pred_bs=256, max_steps=None):
        assert arch in ARCHS
        self.arch = arch
        self.cfg = ARCHS[arch]
        self.block = self.cfg["block"]
        self.budget_s = budget_s
        self.batch, self.lr, self.wd, self.betas = batch, lr, wd, betas
        self.warmup, self.clip = warmup, clip
        self.dev_frac, self.dev_positions = dev_frac, dev_positions
        self.checkpoints = [c for c in checkpoints if c <= budget_s]
        self.seed, self.pred_bs = seed, pred_bs
        self.max_steps = max_steps  # si no es None: presupuesto POR PASOS (reproducible bajo contencion de CPU)
        self.name = f"{arch}-" + ("d%d-L%d-h%d-ctx%d" % (self.cfg["d"], self.cfg["n_layer"], self.cfg["n_head"], self.block)
                                  if arch == "transformer" else "d%d-L%d-ctx%d" % (self.cfg["d"], self.cfg["n_layer"], self.block))
        self.info = {}  # se rellena en fit(); harness.run lo vuelca al JSON via extra=

    # ------------------------------------------------------------------ build / fit
    def _build(self, V):
        c = self.cfg
        if self.arch == "transformer":
            return _Transformer(V, self.block, c["d"], c["n_layer"], c["n_head"])
        return _GRU(V, c["d"], c["n_layer"])

    def fit(self, train, V):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.V = V
        n_dev = int(len(train) * self.dev_frac)
        self.tr = torch.from_numpy(train[:len(train) - n_dev].astype(np.int64))
        self.dev = np.asarray(train[len(train) - n_dev:])  # train-dev: ultimo 5% de TRAIN, nunca en el entrenamiento
        self.net = self._build(V)
        self.net.train()
        self._n_params = sum(p.numel() for p in self.net.parameters())  # tied embedding contado una vez
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.wd)

        N, T, B = len(self.tr), self.block, self.batch
        ar = torch.arange(T)
        t_train, step, last_loss = 0.0, 0, float("nan")
        cks = list(self.checkpoints)
        curve = []
        f0 = time.time()
        while True:
            frac = min(1.0, (step / self.max_steps) if self.max_steps else (t_train / self.budget_s))
            lr = self.lr * min(1.0, (step + 1) / self.warmup) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            t0 = time.perf_counter()
            ix = torch.randint(0, N - T - 1, (B,))
            idx = ix[:, None] + ar[None, :]
            x, y = self.tr[idx], self.tr[idx + 1]
            loss = F.cross_entropy(self.net(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
            opt.step()
            step += 1
            t_train += time.perf_counter() - t0
            last_loss = float(loss.item())
            while cks and t_train >= cks[0]:
                db = self._dev_bpc()
                curve.append(dict(train_s=round(t_train, 2), steps=step, tokens_seen=step * B * T,
                                  dev_bpc=round(db, 4), lr_now=lr, train_loss_last=last_loss))
                cks.pop(0)
                self.net.train()
            if (self.max_steps and step >= self.max_steps) or (not self.max_steps and t_train >= self.budget_s):
                break
        fit_wall = time.time() - f0
        # actualizar IN PLACE: harness.run pasa extra=self.info antes de llamar a fit()
        self.info.clear()
        self.info.update(
            arch=self.arch, name=self.name, n_params=self._n_params, budget_s=self.budget_s,
            steps=step, batch=B, block=T, tokens_seen=step * B * T,
            train_s=round(t_train, 2), fit_wall_incl_dev_s=round(fit_wall, 2),
            train_chars_per_s=step * B * T / t_train,  # caracteres (tokens) de entrenamiento por s de entrenamiento
            train_chars_used=int(N), dev_chars=int(n_dev), epochs_equiv=step * B * T / N,
            final_train_loss_nats=last_loss, curve_dev=curve,
            hparams=dict(lr=self.lr, schedule="warmup %d steps + cosine sobre fraccion de tiempo de pared" % self.warmup,
                         wd=self.wd, betas=self.betas, clip=self.clip, dev_frac=self.dev_frac,
                         dev_positions=self.dev_positions, checkpoints_s=self.checkpoints, seed=self.seed,
                         pred_bs=self.pred_bs, d=self.cfg["d"], n_layer=self.cfg["n_layer"],
                         n_head=self.cfg.get("n_head"), block=T),
        )
        self.net.eval()
        return self

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, ids, pos):
        """p(ids[pos] | ids[max(0,pos-block):pos]). Solo mira ids[:pos] (leak-free).
        Batching: cada contexto se alinea a la izquierda en una ventana de longitud block y se rellena
        a la DERECHA con ceros; como ambas redes son causales, la fila de la ultima posicion real
        no ve el relleno. Se toma el logit en el indice L-1 (L = longitud real del contexto)."""
        self.net.eval()
        ids = np.asarray(ids)
        pos = np.asarray(pos, dtype=np.int64)
        T, V = self.block, self.V
        out = np.empty((len(pos), V), dtype=np.float32)
        ar = np.arange(T)
        for s in range(0, len(pos), self.pred_bs):
            pp = pos[s:s + self.pred_bs]
            L = np.minimum(pp, T)
            assert (L >= 1).all(), "pos debe ser >= 1"
            idx = pp[:, None] - L[:, None] + ar[None, :]
            valid = ar[None, :] < L[:, None]
            X = np.where(valid, ids[np.clip(idx, 0, len(ids) - 1)], 0).astype(np.int64)
            logits = self.net(torch.from_numpy(X))
            last = torch.from_numpy(L - 1)
            lg = logits[torch.arange(len(pp)), last]
            p = torch.softmax(lg.double(), dim=-1).clamp_min(1e-12)
            p = p / p.sum(-1, keepdim=True)
            out[s:s + len(pp)] = p.float().numpy()
        return out

    @torch.no_grad()
    def _dev_bpc(self):
        pos = np.linspace(self.block, len(self.dev) - 1, self.dev_positions).astype(np.int64)
        P = self.predict(self.dev, pos).astype(np.float64)
        p_true = P[np.arange(len(pos)), self.dev[pos]]
        return float(-np.log2(np.maximum(p_true, 1e-12)).mean())

    def n_params(self):
        return int(self._n_params)

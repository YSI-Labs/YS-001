"""GRU de referencia (arquitectura IDENTICA a experiments/r3_gru_paired/model.py: GRU 2 capas, d=256,
ctx 128, batch 64, nn.Embedding/nn.GRU/nn.Linear con inicializacion por defecto de PyTorch) entrenada con
optimizadores learning-rate-free de lrfree_opt.py.

Diferencias respecto al baseline AdamW (deliberadas, para no contar constantes ocultas):
  - sin warmup, sin schedule coseno, sin weight decay, sin clip por defecto (clip=None);
  - presupuesto por NUMERO DE PASOS (2600) igual que el baseline;
  - 2 hilos de CPU (el baseline de r3_gru_paired usaba 6; el resultado puede diferir en la ultima cifra por
    redondeo de coma flotante, no por el metodo);
  - curva de dev interna (ultimo 5% del train) cada `eval_every` pasos; ese tiempo NO cuenta en train_cpu_s.
La secuencia de minibatches es la misma que la del baseline para una misma semilla (mismo orden de RNG).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "2"

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

import sys  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from lrfree_opt import DoG, Prodigy, CoinKT  # noqa: E402


class _GRU(nn.Module):
    def __init__(self, V, d, n_layer):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.gru = nn.GRU(d, d, num_layers=n_layer, batch_first=True)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.head(h)


class LRFreeGRU:
    """Contrato harness: fit(train, V), predict(ids, pos) -> [len(pos), V] normalizado, n_params()."""

    def __init__(self, opt="dog", opt_kw=None, seed=0, steps=2600, batch=64, block=128, d=256, n_layer=2,
                 dev_frac=0.05, dev_positions=2000, eval_every=400, clip=None, pred_bs=256):
        assert opt in ("dog", "prodigy", "coinkt", "adamw")
        self.opt_name, self.opt_kw = opt, dict(opt_kw or {})
        self.seed, self.steps, self.batch, self.block = seed, steps, batch, block
        self.d, self.n_layer = d, n_layer
        self.dev_frac, self.dev_positions, self.eval_every = dev_frac, dev_positions, eval_every
        self.clip, self.pred_bs = clip, pred_bs
        self.name = f"lrfree-{opt}-d{d}-L{n_layer}-ctx{block}"
        self.info = {}

    def _make_opt(self, params):
        if self.opt_name == "dog":
            return DoG(params, **self.opt_kw)
        if self.opt_name == "prodigy":
            return Prodigy(params, **self.opt_kw)
        if self.opt_name == "coinkt":
            return CoinKT(params, **self.opt_kw)
        return torch.optim.AdamW(params, **self.opt_kw)

    def _diag(self):
        g = self.opt.param_groups[0]
        if self.opt_name == "dog":
            return dict(r_bar=g["r_bar"], eta=g["eta"])
        if self.opt_name == "prodigy":
            return dict(d=g["d"], r=g["r"])
        if self.opt_name == "coinkt":
            xs = [self.opt.state[p]["x"].abs().mean().item() for p in self.net.parameters()
                  if "x" in self.opt.state[p]]
            return dict(mean_abs_x=float(np.mean(xs)) if xs else None)
        return {}

    def fit(self, train, V):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.V = V
        n_dev = int(len(train) * self.dev_frac)
        self.tr = torch.from_numpy(train[:len(train) - n_dev].astype(np.int64))
        self.dev = np.asarray(train[len(train) - n_dev:])
        self.net = _GRU(V, self.d, self.n_layer)
        self.net.train()
        self._n_params = sum(p.numel() for p in self.net.parameters())
        self.opt = self._make_opt(self.net.parameters())

        N, T, B = len(self.tr), self.block, self.batch
        ar = torch.arange(T)
        train_cpu = train_wall = 0.0
        curve, diverged_at, last_loss = [], None, float("nan")
        f0 = time.time()
        for step in range(1, self.steps + 1):
            c0, w0 = time.process_time(), time.perf_counter()
            ix = torch.randint(0, N - T - 1, (B,))
            idx = ix[:, None] + ar[None, :]
            x, y = self.tr[idx], self.tr[idx + 1]
            loss = F.cross_entropy(self.net(x).reshape(-1, V), y.reshape(-1))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
            self.opt.step()
            train_cpu += time.process_time() - c0
            train_wall += time.perf_counter() - w0
            last_loss = float(loss.item())
            if not math.isfinite(last_loss):
                diverged_at = step
                break
            if step % self.eval_every == 0 or step == self.steps:
                db = self._dev_bpc()
                entry = dict(step=step, dev_bpc=round(db, 4), train_cpu_s=round(train_cpu, 2),
                             train_loss_last=round(last_loss, 4), **self._diag())
                curve.append(entry)
                print("[progreso]", self.opt_name, entry, flush=True)
                self.net.train()
        fit_wall = time.time() - f0
        self.info.clear()
        self.info.update(
            arch="gru", opt=self.opt_name, name=self.name, n_params=self._n_params, steps=step,
            batch=B, block=T, tokens_seen=step * B * T, train_cpu_s=round(train_cpu, 2),
            train_wall_s=round(train_wall, 2), fit_wall_incl_dev_s=round(fit_wall, 2),
            train_chars_used=int(N), dev_chars=int(n_dev), epochs_equiv=step * B * T / N,
            final_train_loss_nats=last_loss, diverged_at_step=diverged_at, curve_dev=curve,
            hparams=dict(opt=self.opt_name, opt_kw=self.opt_kw, lr="none (learning-rate-free)",
                         schedule="none", wd=0.0, clip=self.clip, momentum=0.9, dev_frac=self.dev_frac,
                         dev_positions=self.dev_positions, seed=self.seed, d=self.d, n_layer=self.n_layer,
                         block=T, batch=B, steps=self.steps, eval_every=self.eval_every),
        )
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, ids, pos):
        """p(ids[pos] | ids[max(0,pos-block):pos]). Solo mira ids[:pos] (leak-free). Igual que el baseline."""
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

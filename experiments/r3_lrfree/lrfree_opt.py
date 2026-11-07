"""Optimizadores learning-rate-free (sin tasa de aprendizaje) para PyTorch.

Tres metodos, todos como torch.optim.Optimizer:

- DoG (Ivgi, Hinder & Carmon 2023, "DoG is SGD's best friend"):
      eta_t = rbar_t / sqrt(sum_{i<=t} ||g_i||^2),   rbar_t = max_{i<=t} ||x_i - x0||,
      rbar_0 = reps * (1 + ||x0||).
  Constantes: reps (inicial, 1e-6 como en el paper), momento beta=0.9 (EMA con correccion de sesgo).
  Escalar GLOBAL (todos los parametros comparten eta y rbar), como en el paper.

- Prodigy (Mishchenko & Defazio 2023), variante tipo Adam, forma simple:
      r_k = b3*r_{k-1} + d_k <g_k, x0 - x_k>,   s_k = b3*s_{k-1} + d_k g_k,   b3 = sqrt(beta2)
      d_{k+1} = max(d_k, r_k / ||s_k||_1)                    (estimador tipo D-Adaptation)
      x_{k+1} = x_k - d_k * lr * bc_k * m_k / (sqrt(v_k) + d_k * eps)
      con m, v = Adam sobre d_k * g_k, bc_k = sqrt(1-beta2^k)/(1-beta1^k).
  Constantes: d0 (inicial, 1e-6 por defecto), betas=(0.9, 0.999), eps=1e-8, lr=1 (convencion del paquete).
  Escalar GLOBAL d.

- CoinKT (apuestas de monedas KT por coordenada; Orabona & Pal 2016, Orabona & Tommasi 2017):
  por coordenada i, con g~ = momento EMA(0.9) de g (correccion de sesgo) y paso t:
      c_t = -clip(g~_t / L_{t-1}, -1, 1)       (moneda con la escala ya conocida; L_0 = 0 -> c=0)
      W_t = W_{t-1} * (1 + beta_t * c_t)        (riqueza; >= 0 porque |beta|<=1, |c|<=1)
      Theta_t = Theta_{t-1} + c_t;  L_t = max(L_{t-1}, |g~_t|)
      beta_{t+1} = Theta_t / (t+1)              (fraccion KT)
      x_{t+1} = beta_{t+1} * W_t / L_t          (desplazamiento apostado)
  El parametro es theta = x0 + x. Constantes: W0 (riqueza inicial por coordenada), momento 0.9.
  Nota: la version con W_t = W_{t-1} - g x (sin recorte de la moneda) diverge (riqueza < 0): ver informe.

Ninguna tasa de aprendizaje ni decaimiento de peso se fija a mano aqui.
Todos guardan x0 (copia de los parametros iniciales) en param_groups, asi que el estado es reproducible.
"""
import math

import torch
from torch.optim import Optimizer

_TINY = 1e-30


def _sq_sum(tensors):
    return sum(float((t.double() ** 2).sum()) for t in tensors)


class DoG(Optimizer):
    """Distance over Gradients, escalar global, con momento EMA de correccion de sesgo."""

    def __init__(self, params, reps=1e-6, momentum=0.9):
        params = list(params)
        super().__init__(params, dict(reps=reps, momentum=momentum))
        group = self.param_groups[0]
        x0 = [p.detach().clone() for p in params]
        group["x0"] = x0
        group["r_bar"] = reps * (1.0 + math.sqrt(_sq_sum(x0)))  # r_eps = reps*(1+||x0||)
        group["G2"] = 0.0
        group["k"] = 0
        group["eta"] = float("nan")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["k"] += 1
            k = group["k"]
            beta = group["momentum"]
            pairs = [(p, x0) for p, x0 in zip(group["params"], group["x0"]) if p.grad is not None]
            group["G2"] += _sq_sum([p.grad for p, _ in pairs])
            if group["G2"] <= 0.0:
                continue
            eta = group["r_bar"] / math.sqrt(group["G2"])
            group["eta"] = eta
            bc = 1.0 - beta ** k
            for p, _ in pairs:
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                st["m"].mul_(beta).add_(p.grad, alpha=1.0 - beta)
                p.add_(st["m"], alpha=-eta / bc)
            dist = math.sqrt(sum(float(((p - x0).double() ** 2).sum()) for p, x0 in
                                 zip(group["params"], group["x0"])))
            group["r_bar"] = max(group["r_bar"], dist)
        return loss


class Prodigy(Optimizer):
    """Prodigy, variante Adam (escalar d global). Ver docstring del modulo."""

    def __init__(self, params, d0=1e-6, betas=(0.9, 0.999), eps=1e-8, lr=1.0):
        params = list(params)
        super().__init__(params, dict(d0=d0, betas=betas, eps=eps, lr=lr))
        group = self.param_groups[0]
        group["x0"] = [p.detach().clone() for p in params]
        group["d"] = float(d0)
        group["r"] = 0.0
        group["k"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            pairs = [(p, x0) for p, x0 in zip(group["params"], group["x0"]) if p.grad is not None]
            group["k"] += 1
            k = group["k"]
            b1, b2 = group["betas"]
            b3 = math.sqrt(b2)
            d = group["d"]
            # numerador: <g, x0 - x> ponderado por d (antes de mover x)
            num = sum(float(torch.sum(p.grad * (x0 - p))) for p, x0 in pairs)
            group["r"] = b3 * group["r"] + d * num
            # denominador: ||s||_1 con s = b3 s + d g
            l1 = 0.0
            for p, _ in pairs:
                st = self.state[p]
                if "s" not in st:
                    st["s"] = torch.zeros_like(p)
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["s"].mul_(b3).add_(p.grad, alpha=d)
                l1 += float(st["s"].abs().sum())
            if l1 > 0.0:
                d = max(d, group["r"] / l1)
            group["d"] = d
            bc = math.sqrt(1.0 - b2 ** k) / (1.0 - b1 ** k)
            dlr = d * group["lr"] * bc
            for p, _ in pairs:
                st = self.state[p]
                st["m"].mul_(b1).add_(p.grad, alpha=d * (1.0 - b1))
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=d * d * (1.0 - b2))
                denom = st["v"].sqrt().add_(d * group["eps"])
                p.addcdiv_(st["m"], denom, value=-dlr)
        return loss


class CoinKT(Optimizer):
    """Apuestas de monedas KT por coordenada (sin escala global, sin LR). Ver docstring del modulo."""

    def __init__(self, params, wealth0=1e-3, momentum=0.9):
        params = list(params)
        super().__init__(params, dict(wealth0=wealth0, momentum=momentum))
        group = self.param_groups[0]
        group["x0"] = [p.detach().clone() for p in params]
        group["k"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            pairs = [(p, x0) for p, x0 in zip(group["params"], group["x0"]) if p.grad is not None]
            group["k"] += 1
            k = group["k"]
            beta = group["momentum"]
            bc = 1.0 - beta ** k
            for p, x0 in pairs:
                st = self.state[p]
                if "W" not in st:
                    st["m"] = torch.zeros_like(p)
                    st["L"] = torch.zeros_like(p)      # escala: max |g~| vista hasta t-1
                    st["Theta"] = torch.zeros_like(p)  # suma de monedas c
                    st["W"] = torch.full_like(p, float(group["wealth0"]))
                    st["beta"] = torch.zeros_like(p)   # fraccion KT usada en el paso actual
                    st["x"] = torch.zeros_like(p)
                st["m"].mul_(beta).add_(p.grad, alpha=1.0 - beta)
                gh = st["m"] / bc
                # moneda normalizada con la escala CONOCIDA antes de apostar (L_{t-1}), recortada a [-1,1]:
                # asi la riqueza W_t = W_{t-1}(1 + beta_t c_t) >= 0 siempre (|beta|<=1, |c|<=1).
                pos_prev = st["L"] > 0
                Lp = st["L"].clamp_min(_TINY)
                c = torch.where(pos_prev, -(gh / Lp).clamp(-1.0, 1.0), torch.zeros_like(gh))
                st["W"].mul_(1.0 + st["beta"] * c)
                st["Theta"].add_(c)
                st["L"] = torch.maximum(st["L"], gh.abs())
                # apuesta para el paso siguiente: x_{t+1} = (Theta_t/(t+1)) * W_t / L_t
                st["beta"] = st["Theta"] / (k + 1)
                Ln = st["L"].clamp_min(_TINY)
                x_new = torch.where(st["L"] > 0, st["beta"] * st["W"] / Ln, torch.zeros_like(gh))
                st["x"] = x_new
                p.copy_(x0 + x_new)
        return loss

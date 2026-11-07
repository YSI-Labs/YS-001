"""DoG (Distance over Gradients, Ivgi et al. 2023 [M: verificar cita]) sin tasa de aprendizaje: eta_t = rbar_t / sqrt(sum_{i<=t} ||g_i||^2), rbar_t = max_i ||x_i - x_0||,
rbar_0 = 1e-6 (1 + ||x_0||). Dos variantes: global (un eta para toda la red) y por capa/tensor (L-DoG, layerwise=True). Momento EMA con correccion de sesgo (beta=0.9, igual que
experiments/r3_lrfree, donde DoG quedo a +0.099 bpc de AdamW en una GRU de 0.84M con 2600 pasos). Sin weight decay ni schedule. Constantes libres: reps=1e-6 y beta=0.9."""
import math, torch

class DoG(torch.optim.Optimizer):
    def __init__(self, params, reps=1e-6, beta=0.9, layerwise=False):
        super().__init__(params, dict(reps=reps, beta=beta, layerwise=layerwise)); self.t = 0; self._init = False; self.rbar_g = None; self.gsq_g = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for grp in self.param_groups:
            ps = [p for p in grp["params"] if p.grad is not None]
            if not self._init:
                for p in ps: st = self.state[p]; st["x0"] = p.detach().clone(); st["m"] = torch.zeros_like(p); st["rbar"] = None; st["gsq"] = 0.0
                n0 = math.sqrt(sum(float((self.state[p]["x0"].float() ** 2).sum()) for p in ps)); self.rbar_g = grp["reps"] * (1.0 + n0); self.gsq_g = 0.0; self._init = True
                for p in ps: self.state[p]["rbar"] = grp["reps"] * (1.0 + float(self.state[p]["x0"].float().norm()))
            self.t += 1; b = grp["beta"]; bc = 1.0 - b ** self.t
            if grp["layerwise"]:
                for p in ps:
                    st = self.state[p]; g = p.grad; st["gsq"] += float((g.float() ** 2).sum()); st["rbar"] = max(st["rbar"], float((p - st["x0"]).float().norm()))
                    eta = st["rbar"] / math.sqrt(max(st["gsq"], 1e-30)); st["m"].mul_(b).add_(g, alpha=1 - b); p.add_(st["m"], alpha=-eta / bc)
            else:
                self.gsq_g += sum(float((p.grad.float() ** 2).sum()) for p in ps); d2 = sum(float(((p - self.state[p]["x0"]).float() ** 2).sum()) for p in ps)
                self.rbar_g = max(self.rbar_g, math.sqrt(d2)); eta = self.rbar_g / math.sqrt(max(self.gsq_g, 1e-30))
                for p in ps: st = self.state[p]; st["m"].mul_(b).add_(p.grad, alpha=1 - b); p.add_(st["m"], alpha=-eta / bc)
            self.last_eta = eta if not grp["layerwise"] else None
        return loss

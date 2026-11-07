"""Corrida final reproducible de Yax Che' (bosque MDL). Configuracion fijada por dev (ver REPORT.md).
Escribe result.json (formato de harness.run, variante B=64) y scan.json (B=1,4,16,64 y profundidad).
Uso: python -I run.py [n_eval]"""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import harness
from model import Yaxche

CFG = dict(B=64, S=1_000_000, L=8, feat_frac=1.0, max_depth=64, seed=0)


class DepthCap:
    """Envoltorio: mismo bosque, pero las hojas se toman a profundidad <= d (ablacion de profundidad)."""
    def __init__(self, m, d):
        self.m, self.d, self.name = m, d, f"{m.name}-depth<={d}"
    def predict(self, ids, pos):
        return self.m.predict(ids, pos, max_depth=self.d)
    def n_params(self):
        return self.m.n_params()


if __name__ == "__main__":
    n_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    train, val, itos = harness.load()
    V = len(itos)
    m = Yaxche(itos, **CFG)
    t0 = time.time()
    m.fit(train, V)
    fit_wall = time.time() - t0
    scan = dict(cfg=CFG, n_eval=n_eval, fit_wall_s=fit_wall, F_build_s=m.F_build_s,
                tree_info=m.tree_info, by_B={})
    for b in (1, 4, 16, 64):
        m.set_B(b)
        r = harness.evaluate(m, val, V, n_eval=n_eval, check_leak=(b == 64))
        scan["by_B"][b] = dict(bpc=r["bpc"], n_params=m.n_params(), eval_chars_per_s=r["eval_chars_per_s"],
                               eval_time_s=r["eval_time_s"], max_norm_err=r["max_norm_err"],
                               min_prob=r["min_prob"], leak_free=r.get("leak_free"),
                               fit_s_prefix=float(sum(m.fit_s_per_tree[:b]) + m.F_build_s))
        print("B", b, round(r["bpc"], 4), m.n_params(), flush=True)
    # profundidad MDL (B=64): caps de profundidad como ablacion
    m.set_B(64)
    scan["depth_ablation"] = {}
    for d in (2, 4, 6, 8, 12, 16, 24, 32, 64):
        r = harness.evaluate(DepthCap(m, d), val, V, n_eval=min(n_eval, 20_000), check_leak=False)
        scan["depth_ablation"][d] = dict(bpc=r["bpc"])
        print("depth", d, round(r["bpc"], 4), flush=True)
    json.dump(scan, open(os.path.join(HERE, "scan.json"), "w"), indent=1, default=str)
    # resultado final: B=64, muestra de texto con el mismo protocolo que harness.run
    prompt = val[1000:1064]
    sample = harness.generate(m, prompt, 400, V, seed=0, itos=itos)
    res = dict(scan["by_B"][64])
    res.update(name=m.name, fit_cpu_s=fit_wall, fit_wall_s=fit_wall, n_params=int(m.n_params()),
               train_chars=len(train), train_chars_per_s=len(train) / max(fit_wall, 1e-9),
               prompt="".join(itos[i] for i in prompt), sample=sample, cfg=CFG)
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1, default=str)
    print("SAMPLE:", res["sample"][:400])
    print("DONE", round(fit_wall, 1), "s fit", flush=True)

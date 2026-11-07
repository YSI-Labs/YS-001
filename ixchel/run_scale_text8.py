"""Prueba de escala en text8 (V=27). Uso: python -I run_scale_text8.py <config> <N_train_chars> [n_eval]
config: A (cadena contigua KN orden<=8, mezcla EM)  |  full (A,W,S + Kb,Kc,WL,W2 + cache 4 ventanas + PoE)"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "text8"
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness
from core_v02 import Ixchel, default_chains
try:
    import psutil
except Exception:
    psutil = None
INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])

def build(cfg):
    d = default_chains()
    if cfg == "A":
        return dict(chains={"A": d["A"]}, gate_chains=("A",), mix="em", cache_windows=())
    if cfg == "full":      # sin cadena L (text8 no tiene saltos de linea)
        ch = {k: d[k] for k in ("A", "W", "S")}; ch.update(LEX)
        return dict(chains=ch, mix="poe", cache_windows=(256, 1024, 4096, INF))
    if cfg == "full_c128":
        ch = {k: d[k] for k in ("A", "W", "S")}; ch.update(LEX)
        return dict(chains=ch, mix="poe", cache_windows=(128,))
    raise SystemExit("config?")

if __name__ == "__main__":
    cfg, N = sys.argv[1], int(sys.argv[2]); n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000
    train, val, itos = harness.load(); V = len(itos)
    train = train[:N]
    m = Ixchel(name=f"Ixchel-t8-{cfg}-{N}", verbose=True, **build(cfg))
    t0 = time.time(); c0 = time.process_time()
    m.fit(train, V)
    fit_wall, fit_cpu = time.time() - t0, time.process_time() - c0
    peak = psutil.Process().memory_info().peak_wset / 2**20 if psutil and hasattr(psutil.Process().memory_info(), "peak_wset") else None
    t1 = time.time()
    res = harness.evaluate(m, val, V, n_eval=n_eval, check_leak=False)
    res["leak_free"] = bool(harness.leak_check(m, val[:300_000], V, n=16))
    res.update(cfg=cfg, n_train=int(N), n_params=int(m.n_params()), fit_wall_s=fit_wall, fit_cpu_s=fit_cpu,
               peak_rss_mb=peak, train_chars_per_s=N / fit_wall, n_eval=n_eval, dev_bpc_mix=m.dev_bpc_mix,
               poe_w=None if m.poe_w is None else m.poe_w.tolist(), expert_names=m.expert_names())
    sample = harness.generate(m, val[1000:1064], 400, V, seed=0, itos=itos)
    res["sample"] = sample
    out = os.path.join(HERE, "results", f"t8_{cfg}_{N}.json"); json.dump(res, open(out, "w"), indent=1)
    print(f"[t8 {cfg} N={N:,}] bpc={res['bpc']:.4f} params={res['n_params']:,} fit={fit_wall:.0f}s (cpu {fit_cpu:.0f}s) "
          f"peak={peak} MB leak_free={res['leak_free']} eval={time.time()-t1:.0f}s", flush=True)
    print(sample[:400])

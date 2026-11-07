"""Entrena UNA VEZ Ixchel v0.3 (config 'v03' de ixchel/run_v02.py) sobre Shakespeare y guarda las tablas
empaquetadas en model_v03/ (arrays .npy sin comprimir -> se cargan con mmap y se comparten entre procesos).
Uso: python -I train_save.py   (dura ~1-3 min en 2 hilos)."""
import os
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
from core_v02 import Ixchel, default_chains, build_char_tables
from engine import pack_model

INF = 1 << 40
LEX = dict(
    Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
    Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
    WL=[[], ["wlen"], ["wlen", "c1", "c2"]],
    W2=[[], ["wp"], ["wp", "pw2"]])
CHAINS = {**default_chains(), **LEX}           # copia literal de run_v02.CFG['v03']
CACHE_WINDOWS = (256, 1024, 4096, INF)

if __name__ == "__main__":
    train, val, itos = harness.load()
    V = len(itos)
    m = Ixchel(name="Ixchel-v0.3-engine", chains=CHAINS, cache_windows=CACHE_WINDOWS, mix="poe", verbose=True)
    t0 = time.time()
    m.fit(train, V)
    fit_s = time.time() - t0
    print(f"fit {fit_s:.1f}s  dev_bpc_mix={m.dev_bpc_mix:.4f}  pares={sum(len(t['psym']) for t in m.tabs.values()):,}", flush=True)
    out = os.path.join(HERE, "model_v03")
    cls, isl, isn = build_char_tables(itos)
    meta = dict(name="Ixchel-v0.3", V=V, itos=itos, chains=CHAINS, cache_windows=list(CACHE_WINDOWS),
                poe_w=[float(x) for x in m.poe_w], fit_s=fit_s, dev_bpc_mix=m.dev_bpc_mix,
                train_chars=int(len(train)), source="ixchel/run_v02.py CFG['v03'] (copia en r3_engine)",
                dev_bpc_experts=m.dev_bpc_experts)
    pack_model(out, m.tabs, m.base, meta, cls, isl, isn)
    print("guardado en", out, flush=True)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("itos", "chains", "dev_bpc_experts")}, indent=1))

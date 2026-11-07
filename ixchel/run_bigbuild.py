"""Construye (fuera de memoria) las tablas de Ixchel v0.4 sobre big1. Uso: python -I run_bigbuild.py <grupo 0|1|2> [chunk]
Cada grupo construye un subconjunto de cadenas; se lanzan los 3 en paralelo. Salida: data/big1_model/<cadena>/*.npy"""
import os, sys, time, json
os.environ["IXCHEL_DATA"] = "big1"; os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np
from core_v04 import Ixchel, default_chains
from bigbuild import build_chains_big
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
GROUPS = [["A", "W2", "WL"], ["S", "W", "L"], ["Kb", "Kc"]]
if __name__ == "__main__":
    g = int(sys.argv[1]); chunk = int(float(sys.argv[2])) if len(sys.argv) > 2 else 12_000_000
    train = np.load(os.path.join(ROOT, "data", "big1", "train.npy"), mmap_mode="r"); n = len(train); cut = int(n * 0.95)
    m = Ixchel(name="big-build", verbose=False, chains={**default_chains(), **LEX}, mix="poe", cache_windows=())
    out = os.path.join(ROOT, "data", "big1_model")
    t = time.time(); meta = build_chains_big(m, train, cut, out, nparts=32, chunk=chunk, chains=GROUPS[g], verbose=True)
    print("BUILD_DONE", g, json.dumps(meta["_stats"]), "wall", round(time.time() - t), flush=True)

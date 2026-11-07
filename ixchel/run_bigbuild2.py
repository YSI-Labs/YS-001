"""Construye (fuera de memoria) las tablas de Ixchel v0.5 sobre big2 (corpus corregido y reequilibrado). Uso: python -I run_bigbuild2.py <grupo 0..3> [chunk]
Salida: data/big2_model/<cadena>/*.npy"""
import os, sys, time, json
os.environ["IXCHEL_DATA"] = "big2"; os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np
from core_v05 import Ixchel
from chains_v05 import v05_chains
from bigbuild import build_chains_big
GROUPS = [["A", "WL"], ["W", "CB", "L", "CK"], ["S", "W2", "CI", "CF"], ["Kb", "Kc", "CS", "CSK"]]
if __name__ == "__main__":
    g = int(sys.argv[1]); chunk = int(float(sys.argv[2])) if len(sys.argv) > 2 else 12_000_000
    train = np.load(os.path.join(ROOT, "data", "big2", "train.npy"), mmap_mode="r")
    man = json.load(open(os.path.join(ROOT, "data", "big2", "manifest.json"))); cut = int(man["main_chars_with_repetition"])
    m = Ixchel(name="big2-build", verbose=False, chains=v05_chains(), mix="poe", cache_windows=())
    out = os.path.join(ROOT, "data", "big2_model")
    t = time.time(); meta = build_chains_big(m, train, cut, out, nparts=32, chunk=chunk, chains=GROUPS[g], verbose=True)
    print("BUILD_DONE", g, json.dumps(meta["_stats"]), "wall", round(time.time() - t), flush=True)

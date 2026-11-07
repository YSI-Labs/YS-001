"""Anade model_v03/HK.npy (indice hash de niveles sobre GCK) a un modelo ya empaquetado (sin reentrenar)."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from engine import build_level_hash
M = os.path.join(HERE, "model_v03")
GCK = np.load(os.path.join(M, "GCK.npy")); GCOFF = np.load(os.path.join(M, "GCOFF.npy"))
HK = build_level_hash(GCK, GCOFF).astype(np.int32)
np.save(os.path.join(M, "HK.npy"), HK)
print("HK", HK.shape, HK.nbytes / 2**20, "MB; ocupadas", int((HK >= 0).sum()), "de", len(GCK))

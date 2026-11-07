"""Config Ixchel v0.3 (copia de ixchel/run_v02.py CFG['v03']). Sin cambios de hiperparametros."""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core_v02 import Ixchel, default_chains
INF = 1 << 40
LEX = dict(
    Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
    Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
    WL=[[], ["wlen"], ["wlen", "c1", "c2"]],
    W2=[[], ["wp"], ["wp", "pw2"]])
CFG_V03 = dict(chains={**default_chains(), **LEX}, cache_windows=(256, 1024, 4096, INF), mix="poe")

def make(name, **kw):
    return Ixchel(name=name, verbose=True, **{**CFG_V03, **kw})

def bpc_of_logp(lp, x):
    """lp: [n,V] log-probs (float), x: [n] simbolos -> bits por caracter medio."""
    import numpy as np
    return float(-(lp[np.arange(len(x)), x].astype(np.float64)).mean() / np.log(2))

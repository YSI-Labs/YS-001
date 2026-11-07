import sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'experiments', 'r1_chan'))
import numpy as np, harness
from model import ChanHDC
train, val, itos = harness.load(); V = len(itos)
small = train[:500_000]
for m in [ChanHDC(D=1024, K=4, mode="wb", H=31, zthr=2.0)]:
    t0 = time.time(); m.fit(small, V); print(m.name, "fit", round(time.time()-t0,2))
    res = harness.evaluate(m, val, V, n_eval=2000)
    print({k: res[k] for k in ("bpc","max_norm_err","min_prob","leak_free")})

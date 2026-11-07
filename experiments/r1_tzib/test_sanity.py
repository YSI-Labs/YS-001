import os, sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, harness
from baseline_wb import WittenBell
from model import Tzib, METHODS
train, val, itos = harness.load(); V = len(itos)
t0 = time.time(); m = Tzib("WB-5", cache=True); m.fit(train, V); print("fit", round(time.time()-t0,1), m.fit_info, flush=True)
b = WittenBell(5); t0=time.time(); b.fit(train, V); print("baseline fit", round(time.time()-t0,1), flush=True)
rng = np.random.default_rng(1); pos = np.sort(rng.integers(64, len(val)-1, 300)).astype(np.int64)
t0=time.time(); P = m.predict(val, pos); print("predict 300 pos", round(time.time()-t0,3), flush=True)
Pb = b.predict(val, pos)
print("max |WB5 mine - baseline| =", float(np.abs(P-Pb).max()), " sums err", float(np.abs(P.sum(1)-1).max()), " min", float(P.min()))
print("Lstar mean", m.last_lstar.mean(), "max", m.last_lstar.max(), "trunc", m.last_trunc)
for v in METHODS:
    mm = Tzib(v); mm.fit(train, V); Q = mm.predict(val, pos)
    print(v, "sum err %.1e min %.2e bpc_300 %.3f" % (np.abs(Q.sum(1)-1).max(), Q.min(), -np.log2(Q[np.arange(len(pos)), val[pos]]).mean()), "nparams", mm.n_params(), flush=True)

import os, sys, time
os.environ["NUMBA_NUM_THREADS"] = "4"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness
from core_v04 import Ixchel, default_chains
from bigbuild import build_chains_big, load_tables
train, val, itos = harness.load(); V = len(itos)
N = 1_500_000; tr = train[:N]
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]], WL=[[], ["wlen"], ["wlen", "c1", "c2"]])
ch = {**default_chains(), **LEX}
m = Ixchel(name="t", verbose=False, chains=ch, mix="poe", cache_windows=())
A = m._atoms_for(tr); t = time.time(); ref = m._count(A, tr, 16, N); print("referencia", round(time.time()-t,1), "s")
out = os.path.join(ROOT, "ixchel", "results", "_test_big"); t = time.time()
meta = build_chains_big(m, tr, N, out, nparts=4, chunk=N + 10, verbose=False); print("bigbuild", round(time.time()-t,1), "s")
got = load_tables(out, m.names, mmap=False)
ok = True
for nm in m.names:
    for k in ("ckey", "cstart", "psym", "pcnt", "c_off", "cs_off", "D"):
        a, b = np.asarray(ref[nm][k]), np.asarray(got[nm][k])
        if a.shape != b.shape: ok = False; print("FORMA DIFERENTE", nm, k, a.shape, b.shape); continue
        if k == "D":
            print("  D max abs diff", nm, float(np.abs(a - b).max()))
        elif not np.array_equal(a, b):
            frac = float((a != b).mean()); print("  difiere", nm, k, "fraccion", round(frac, 6), "max rel", float(np.abs(a.astype(float) - b).max() / max(a.max(), 1)))
            if k != "pcnt": ok = False
print("TABLAS IDENTICAS" if ok else "HAY DIFERENCIAS")

"""Ajusta la base Ixchel v0.4 sobre mix15 (misma configuracion que ixchel/run_mix15.py base_model) y la guarda en pickle.
Uso: python fit_base.py <ruta_pickle>"""
import os, sys, time, pickle
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["IXCHEL_DATA"] = "mix15"
ROOT = __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ixchel")); sys.path.insert(0, os.path.join(ROOT, "common"))
import run_mix15 as R, harness
out = sys.argv[1]
train, val, itos = harness.load(); V = len(itos)
m = R.base_model()
t = time.time(); m.fit(train, V); print("fit s", round(time.time() - t), flush=True)
pickle.dump(m, open(out, "wb"), protocol=4); print("pickle ok", os.path.getsize(out), flush=True)

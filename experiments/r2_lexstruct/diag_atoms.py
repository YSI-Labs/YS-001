import os, sys, json, importlib.util
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core
spec = importlib.util.spec_from_file_location("orig_core", os.path.join(ROOT, "ixchel", "core.py"))
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)

train = np.load(os.path.join(ROOT, "data", "shakespeare", "train.npy"))
itos = json.load(open(os.path.join(ROOT, "data", "shakespeare", "vocab.json")))
ids = train[:400_000]
# 1) atomos v0.1 identicos
cols_old = np.arange(28, dtype=np.int64)
cls, isl, isn = orig.build_char_tables(itos)
A_old = np.zeros((len(ids), 28), np.int64); orig._atoms(ids, cls, isl, isn, cols_old, A_old)
cf = core.build_case_flags(itos)
cols_new = np.arange(36, dtype=np.int64)
A_new = np.zeros((len(ids), 36), np.int64)
core._atoms(ids, cls, isl, isn, cf, cols_new, A_new)
print("atomos v0.1 identicos:", bool(np.array_equal(A_old, A_new[:, :28])))

# 2) estadisticas de atomos nuevos
names = {v: k for k, v in core.ATOM_NAMES.items()}
spk = A_new[:, 28]
print("posiciones con hablante asignado:", float((spk != 0).mean()).__round__(4))
hdr = []
ch = [itos[i] for i in ids[:3000]]
txt = "".join(ch)
print("ejemplo texto:", repr(txt[:200]))
for name in ["spos", "plend", "pl1f", "pls", "lsyl", "wcs", "pwcs"]:
    j = core.ATOM_NAMES[name]
    v, c = np.unique(A_new[:, j], return_counts=True)
    top = sorted(zip(c, v), reverse=True)[:8]
    print(name, "top valores:", [(int(vv), round(cc / len(ids), 3)) for cc, vv in top])
# distintos hablantes
print("hablantes distintos (hash>>2):", len(np.unique(spk)))
# silabas de linea vs longitud de linea (sanidad: deberia estar cerca de 10 en pentametros)
pls = A_new[:, 32]
print("media pls (silabas linea previa, tope 15):", float(pls.mean()).__round__(2))

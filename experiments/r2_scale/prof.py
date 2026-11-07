"""Perfil y throughput de Ixchel v0.1 (AWSL). Uso: python -I prof.py <modulo: core|core_opt> <hilos>
Mide: fit completo; conteo (_count) con desglose; inferencia por lote de 4096 posiciones (eval real) y
prediccion de 1 posicion (como en generacion). Guarda results/prof_<modulo>_<hilos>.json y
results/probs_<modulo>_<hilos>.npy (primer lote, para comparar exactitud)."""
import os, sys, time, json
THREADS = sys.argv[2] if len(sys.argv) > 2 else "2"
os.environ["NUMBA_NUM_THREADS"] = THREADS
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import psutil
import harness
import importlib
modname = sys.argv[1] if len(sys.argv) > 1 else "core"
mod = importlib.import_module(modname)
import numba
numba.set_num_threads(int(THREADS))

train, val, itos = harness.load()
V = len(itos)
allc = mod.default_chains()
chains = {k: allc[k] for k in "AWSL"}
m = mod.Ixchel(chains=chains, gate_chains=("A", "W"), gated=True, name="prof", verbose=False)

# warm-up JIT con subconjunto pequeno (no cuenta en tiempos)
m.fit(train[:200_000], V)
t0 = time.time()
m.fit(train, V)
fit_wall = time.time() - t0
out = dict(module=modname, threads=int(THREADS), fit_wall_s=fit_wall, n_params=m.n_params(),
           peak_rss_mb=psutil.Process().memory_info().peak_wset / 2**20)

# ---- conteo desglosado sobre todo el train (mismas llamadas que _count)
n = len(train)
tt = {}
def T(k, t): tt[k] = tt.get(k, 0.0) + (time.time() - t)
t = time.time(); A = m._atoms_for(train); T("atoms", t)
pos = np.arange(m.t0, n, dtype=np.int64)
xs = train[m.t0:n]
t_tab = 0.0; t_H = 0.0; t_cat = 0.0
for nm in m.names:
    t = time.time(); H = m._H(A, pos, nm); t_H += time.time() - t
    t = time.time(); tb = mod.build_chain_tables(H, xs); t_tab += time.time() - t
    t = time.time(); mod._concat_tables(tb); t_cat += time.time() - t
    del H, tb
tt["hashes"] = t_H; tt["build_tables(unique+make)"] = t_tab; tt["concat+index"] = t_cat
tt["count_total"] = sum(tt[k] for k in ("atoms", "hashes", "build_tables(unique+make)", "concat+index"))
out["count_stages_s"] = {k: round(v, 3) for k, v in tt.items()}
out["count_chars_per_s"] = n / tt["count_total"]
del A

# ---- inferencia: lote de 4096 posiciones como en la evaluacion
pos_eval = np.linspace(64, len(val) - 1, 20_000).astype(np.int64)
batch = pos_eval[:4096]
P0 = m.predict(val, batch)  # calentamiento
np.save(os.path.join(HERE, "results", f"probs_{modname}_{THREADS}.npy"), P0)
reps = 5
ts = []
for _ in range(reps):
    t = time.time(); m.predict(val, batch); ts.append(time.time() - t)
out["predict_batch4096_ms_median"] = float(np.median(ts) * 1000)
out["predict_batch4096_pos_per_s"] = 4096 / float(np.median(ts))
# desglose
ta = []; td = []
for _ in range(reps):
    t = time.time()
    if hasattr(m, "_atoms_pos"):
        A = m._atoms_pos(val, batch)
        pp = np.arange(len(batch), dtype=np.int64)
    else:
        A = m._atoms_for(val); pp = batch
    ta.append(time.time() - t)
    t = time.time(); m._chain_dists(m.tabs, A, pp, V); td.append(time.time() - t)
out["predict_stage_atoms_ms"] = float(np.median(ta) * 1000)
out["predict_stage_chain_dists_ms"] = float(np.median(td) * 1000)

# ---- prediccion de 1 posicion (generacion): ids = buffer creciente
buf = val[:4000].copy()
lat = {}
for L in (64, 300, 1000, 3000):
    ts = []
    for _ in range(20):
        t = time.time(); m.predict(buf, np.array([L], dtype=np.int64)); ts.append(time.time() - t)
    lat[L] = float(np.median(ts) * 1000)
out["predict_1pos_ms_by_context_len"] = lat
out["predict_1pos_chars_per_s_ctx300"] = 1000.0 / lat[300]

json.dump(out, open(os.path.join(HERE, "results", f"prof_{modname}_{THREADS}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

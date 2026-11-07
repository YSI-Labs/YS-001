"""Latencia de busqueda en tablas grandes sinteticas (1 par por contexto = peor caso de memoria).
Uso: python -I synth_lat.py <N_contextos> ; escribe results/synth_<N>.json
Mide ns por consulta: (a) busqueda binaria sobre ckey ordenada (como core.py), (b) indice hash abierto (core_opt).
Consultas: 50% claves presentes, 50% ausentes, aleatorias. 1 hilo (la busqueda es memory-bound por nucleo)."""
import os, sys, time, json
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import numba as nb
import psutil
import core_opt as co

N = int(sys.argv[1])
rng = np.random.default_rng(0)
t = time.time()
ckey = np.arange(N, dtype=np.uint64)
ckey <<= np.uint64(20)                       # crecientes y unicas: i*2^20 + r
ckey |= rng.integers(0, 2**20, size=N, dtype=np.uint64)
cstart = np.arange(N + 1, dtype=np.int64)
psym = np.zeros(N, np.uint8)
pcnt = np.ones(N, np.int32)
build_s = time.time() - t
t = time.time()
Mi = 2
while Mi < 2 * N:                             # carga <= 0.5 en el indice
    Mi *= 2
idx = co._build_index(ckey, Mi)
idx_s = time.time() - t
mem_base = (ckey.nbytes + cstart.nbytes + psym.nbytes + pcnt.nbytes)
mem_idx = idx.nbytes
Q = 2_000_000
qi = rng.integers(0, N, size=Q // 2)
qp = (ckey[qi] >> np.uint64(0)).astype(np.uint64)
qa = rng.integers(0, 2**63, size=Q // 2, dtype=np.uint64) >> np.uint64(6)  # ausentes (casi seguro)
Qk = np.concatenate([qp, qa])
rng.shuffle(Qk)


@nb.njit(cache=True)
def bsearch_all(keys, ck):
    hits = 0
    for q in range(keys.shape[0]):
        k = keys[q]
        a = 0
        b = ck.shape[0]
        while a < b:
            mid = (a + b) >> 1
            if ck[mid] < k:
                a = mid + 1
            else:
                b = mid
        if a < ck.shape[0] and ck[a] == k:
            hits += 1
    return hits


@nb.njit(cache=True)
def hash_all(keys, ck, idx, mask):
    hits = 0
    for q in range(keys.shape[0]):
        k = keys[q]
        h = np.int64(co._mix(k) & mask)
        while True:
            j = idx[h]
            if j < 0:
                break
            if ck[j] == k:
                hits += 1
                break
            h = (h + 1) & np.int64(mask)
    return hits


bsearch_all(Qk[:1000], ckey)  # compila
hash_all(Qk[:1000], ckey, idx, np.uint64(Mi - 1))
reps = 3
tb, th = [], []
for _ in range(reps):
    t = time.time(); hb = bsearch_all(Qk, ckey); tb.append(time.time() - t)
    t = time.time(); hh = hash_all(Qk, ckey, idx, np.uint64(Mi - 1)); th.append(time.time() - t)
out = dict(N_contexts=N, bytes_base=int(mem_base), bytes_index=int(mem_idx), bytes_per_ctx_base=mem_base / N,
           build_s=build_s, index_build_s=idx_s, queries=Q,
           bsearch_ns_per_query=float(np.median(tb) / Q * 1e9), hash_ns_per_query=float(np.median(th) / Q * 1e9),
           hits_bsearch=int(hb), hits_hash=int(hh), rss_mb=psutil.Process().memory_info().peak_wset / 2**20)
print(json.dumps(out))
json.dump(out, open(os.path.join(HERE, "results", f"synth_{N}.json"), "w"), indent=1)

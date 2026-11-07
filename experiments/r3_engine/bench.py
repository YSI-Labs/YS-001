"""Benchmarks del motor IxchelEngine (v0.3). Salida: bench.json (todas las cifras del informe salen de aqui).
  python -I bench.py main          -> tokens/s por hilo, latencias p50/p99, memoria, scoring por lotes, referencia O(L^2)
  python -I bench.py worker L N    -> (usado por 'conc') genera N tokens con historial L y escribe JSON por stdout
  python -I bench.py conc          -> escalado con 1,2,4,8 procesos concurrentes (cada uno un hilo numba)
Maquina compartida con otros agentes: tiempos contaminados (se anota en el informe)."""
import os, sys, json, time, ctypes
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
os.environ["NUMBA_NUM_THREADS"] = "1" if MODE in ("worker", "conc") else "2"
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess
import numpy as np
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
from engine import IxchelEngine, Session, _sample
MODEL = os.path.join(HERE, "model_v03")


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def rss_mb():
    try:
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 2**20
    except Exception:
        return float("nan")


def pct(a, q):
    return float(np.percentile(a, q))


def worker(L, N, seed=1, N_lat=1000):
    """N tokens con bucle completo en numba (throughput del kernel) + N_lat tokens con cronometro por token
    desde Python (latencia p50/p99; incluye el despacho Python de cada token)."""
    _, val, itos = harness.load()
    eng = IxchelEngine.load(MODEL)
    eng.generate_fast(eng.new_session(val[:200]), 20, seed=0)   # JIT + paginas del mmap
    eng.generate(eng.new_session(val[:200]), 20, seed=0)
    t0 = time.perf_counter()
    sess = eng.new_session(val[:L]) if L > 0 else eng.new_session()
    t_prompt = time.perf_counter() - t0
    t0 = time.perf_counter()
    eng.generate_fast(sess, N, seed=seed)
    wall = time.perf_counter() - t0
    sess2 = eng.new_session(val[:L]) if L > 0 else eng.new_session()
    tim = np.zeros(N_lat, np.int64)
    t0 = time.perf_counter()
    eng.generate(sess2, N_lat, seed=seed, timings=tim)
    wall2 = time.perf_counter() - t0
    return dict(L=L, N=N, kernel_tok_per_s=N / wall, kernel_us_per_tok=wall / N * 1e6, prompt_s=t_prompt,
                py_loop_tok_per_s=N_lat / wall2, p50_us=pct(tim, 50) / 1e3, p99_us=pct(tim, 99) / 1e3,
                mean_us=float(tim.mean()) / 1e3, N_lat=N_lat)


if __name__ == "__main__":
    if MODE == "worker":
        print(json.dumps(worker(int(sys.argv[2]), int(sys.argv[3]))))
        sys.exit(0)

    if MODE == "conc":
        out = {}
        L, N = 1000, 4000
        for P in (1, 2, 4, 8):
            t0 = time.time()
            procs = [subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), "worker", str(L), str(N)],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for _ in range(P)]
            rows = []
            for p in procs:
                so, _ = p.communicate()
                rows.append(json.loads(so.strip().splitlines()[-1]))
            wall = time.time() - t0
            agg = sum(r["kernel_tok_per_s"] for r in rows)
            out[str(P)] = dict(procs=P, ctx=L, per_proc_kernel_tok_s=[round(r["kernel_tok_per_s"], 1) for r in rows],
                               mean_per_proc_kernel_tok_s=float(np.mean([r["kernel_tok_per_s"] for r in rows])),
                               aggregate_kernel_tok_s=agg, p50_us_mean=float(np.mean([r["p50_us"] for r in rows])),
                               p99_us_mean=float(np.mean([r["p99_us"] for r in rows])), launch_wall_s=wall)
            print("conc", P, out[str(P)], flush=True)
        json.dump(out, open(os.path.join(HERE, "bench_conc.json"), "w"), indent=1)
        sys.exit(0)

    # ------------------------------------------------------------ MAIN
    res = {"machine": f"{os.cpu_count()} hilos logicos; Windows 11; numba {__import__('numba').__version__}; "
                      f"numpy {np.__version__}"}
    _, val, itos = harness.load()
    V = len(itos)
    rss0 = rss_mb()
    t0 = time.perf_counter()
    eng = IxchelEngine.load(MODEL)
    t_load = time.perf_counter() - t0
    res["model"] = dict(load_s_mmap=t_load, model_bytes=eng.model_nbytes(), model_MB=eng.model_nbytes() / 2**20,
                        n_pairs=int(len(eng.A["GPS"])), bytes_per_pair=eng.model_nbytes() / len(eng.A["GPS"]),
                        rss_after_load_MB=rss_mb(), rss_start_MB=rss0)
    # calentamiento y primera pasada (paginas del mmap)
    s = eng.new_session(val[:300]); eng.step(s)
    res["model"]["rss_after_warm_MB"] = rss_mb()

    # ---- throughput y latencia por contexto (1 hilo, proceso propio por medicion en 'worker' para aislar)
    res["gen_by_context"] = {}
    for L in (0, 1000, 10000, 100000):
        # medicion en proceso aislado para que cada L arranque limpio
        p = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "worker", str(L), "3000"],
                           capture_output=True, text=True)
        row = json.loads(p.stdout.strip().splitlines()[-1])
        # memoria de la sesion tras el prompt (en proceso)
        s = eng.new_session(val[:L]) if L > 0 else eng.new_session()
        row["session_bytes_after_prompt"] = s.nbytes()
        row["session_hash_slots"] = int(len(s.hkeys)); row["session_used_keys"] = int(s.st[7])
        res["gen_by_context"][str(L)] = row
        print("gen", L, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    # ---- memoria por sesion (historial creciente): hist + tabla hash
    res["session_memory"] = {}
    for L in (0, 1000, 10000, 100000, 500000):
        s = eng.new_session(val[:L]) if L > 0 else eng.new_session()
        res["session_memory"][str(L)] = dict(bytes=s.nbytes(), MB=s.nbytes() / 2**20, used_keys=int(s.st[7]),
                                            slots=int(len(s.hkeys)))
    print("session memory", res["session_memory"], flush=True)

    # ---- referencia: coste por token del nucleo (predict recalcula todo el prefijo) a distintos L
    res["reference_predict_1pos"] = {}
    from core_v02 import Ixchel
    sys.path.insert(0, HERE)
    import verify as V_
    ref, _ = V_.reference_from_pack(MODEL)
    for L in (1000, 10000, 100000):
        reps = 3 if L < 100000 else 1
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); ref.predict(val[:L + 1], np.array([L], np.int64)); ts.append(time.perf_counter() - t0)
        res["reference_predict_1pos"][str(L)] = dict(s_per_call=float(np.median(ts)), reps=reps)
        print("ref predict L", L, res["reference_predict_1pos"][str(L)], flush=True)

    # ---- scoring por lotes: motor (una pasada, distribucion en todas las posiciones) vs nucleo por lotes
    n_sc = 100_000
    posA = np.arange(64, 64 + n_sc, dtype=np.int64)
    eng.score_positions(val[:2000], np.arange(64, 1000, dtype=np.int64))   # compilacion fuera de la medida
    t0 = time.perf_counter(); Pe = eng.score_positions(val[:64 + n_sc], posA); te = time.perf_counter() - t0
    res["scoring"] = dict(engine_pos=n_sc, engine_s=te, engine_pos_per_s=n_sc / te,
                          engine_bpc_check=float(-np.log2(np.maximum(Pe[np.arange(n_sc), val[posA]].astype(np.float64), 1e-12)).mean()))
    pb = np.linspace(64, len(val) - 1, 4096).astype(np.int64)
    ts = []
    for _ in range(2):
        t0 = time.perf_counter(); ref.predict(val, pb); ts.append(time.perf_counter() - t0)
    res["scoring"]["nucleo_batch4096_s_median"] = float(np.median(ts))
    res["scoring"]["nucleo_batch4096_pos_per_s"] = 4096 / float(np.median(ts))
    res["scoring"]["note"] = ("nucleo: predict por lotes de 4096 posiciones repartidas en val (un solo predict "
                              "reconstruye el indice del cache sobre todo el prefijo); motor: pasada secuencial "
                              "con distribucion en cada posicion de val[:64+100000].")
    print("scoring", res["scoring"], flush=True)
    res["rss_end_MB"] = rss_mb()
    json.dump(res, open(os.path.join(HERE, "bench.json"), "w"), indent=1, default=float)
    print("OK bench.json")

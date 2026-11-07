"""r3_longcache runner. Uso: python -I run_lc.py <cfg> <N_train> [n_eval=20000] [data=text8]
Guarda NLL por posicion (npy) para comparacion pareada y un JSON con bpc, dev por experto, pesos PoE, tiempos."""
import os, sys, json, time
DATA = sys.argv[4] if len(sys.argv) > 4 else "text8"
os.environ["IXCHEL_DATA"] = DATA
os.environ["NUMBA_NUM_THREADS"] = "2"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness
from core_lc import Ixchel, default_chains
from longcache import KeyedWin
try:
    import psutil
except Exception:
    psutil = None

INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
d = default_chains()
FULL_CH = {k: d[k] for k in ("A", "W", "S")}; FULL_CH.update(LEX)          # = run_scale 'full'
BASE = dict(chains=FULL_CH, mix="poe", cache_windows=(256, 1024, 4096, INF))  # base medida por ronda 2/3
CH = lambda k: [("c", j) for j in range(1, k + 1)]

CFG = {
    "base": dict(BASE),
    # (1) cache de palabras: contexto wp y (pw1,wp), ventanas duras
    "wc": dict(BASE, xperts=[KeyedWin("WC", [("w",), ("w2",)], windows=[1024, 4096, 16384, INF])]),
    # (3) tema: contexto wp solo, ventanas 2000 y 5000 (repeticion de palabras del documento reciente)
    "topic": dict(BASE, xperts=[KeyedWin("TP", [("w",)], windows=[2000, 5000])]),
    # (2) recencia: decaimiento exponencial exacto, orden 1..4 de caracteres, tau por dev-ML
    "dec": dict(BASE, xperts=[KeyedWin("DC", CH(4), mode="dec")]),
    # (4) orden maximo del cache: reemplaza el cache WB de orden<=4 por orden<=6 / <=8 (mismas ventanas)
    "ord6": dict(chains=FULL_CH, mix="poe", cache_windows=(), xperts=[KeyedWin("C6", CH(6), windows=[256, 1024, 4096, INF])]),
    "ord8": dict(chains=FULL_CH, mix="poe", cache_windows=(), xperts=[KeyedWin("C8", CH(8), windows=[256, 1024, 4096, INF])]),
    # confirmacion en Shakespeare: base = configuracion v0.3 (chains default+LEX con cadena L), cache WB 4 ventanas
    "sbase": dict(chains={**default_chains(), **LEX}, mix="poe", cache_windows=(256, 1024, 4096, INF)),
    "swc": dict(chains={**default_chains(), **LEX}, mix="poe", cache_windows=(256, 1024, 4096, INF),
                 xperts=[KeyedWin("WC", [("w",), ("w2",)], windows=[1024, 4096, 16384, INF])]),
    # combinacion: orden 6 (reemplaza cache WB<=4) + cache de palabras (ventanas duras)
    "ord6wc": dict(chains=FULL_CH, mix="poe", cache_windows=(),
                  xperts=[KeyedWin("C6", CH(6), windows=[256, 1024, 4096, INF]), KeyedWin("WC", [("w",), ("w2",)], windows=[1024, 4096, 16384, INF])]),
    "ord8wc": dict(chains=FULL_CH, mix="poe", cache_windows=(),
                  xperts=[KeyedWin("C8", CH(8), windows=[256, 1024, 4096, INF]), KeyedWin("WC", [("w",), ("w2",)], windows=[1024, 4096, 16384, INF])]),
    "sord8": dict(chains={**default_chains(), **LEX}, mix="poe", cache_windows=(),
                   xperts=[KeyedWin("C8", CH(8), windows=[256, 1024, 4096, INF])]),
}

if __name__ == "__main__":
    cfg, N = sys.argv[1], int(sys.argv[2])
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000
    train, val, itos = harness.load(); V = len(itos)
    train = train[:N]
    m = Ixchel(name=f"Ixchel-lc-{cfg}-{N}", verbose=True, **CFG[cfg])
    t0, c0 = time.time(), time.process_time()
    m.fit(train, V)
    fit_wall, fit_cpu = time.time() - t0, time.process_time() - c0
    pos = harness.eval_positions(val, n_eval)
    t1 = time.time()
    nll = np.zeros(len(pos))
    for s in range(0, len(pos), 4096):
        pp = pos[s:s + 4096]
        P = m.predict(val, pp).astype(np.float64)
        assert P.shape == (len(pp), V)
        nll[s:s + len(pp)] = -np.log2(np.maximum(P[np.arange(len(pp)), val[pp]], 1e-12))
    eval_wall = time.time() - t1
    leak = bool(harness.leak_check(m, val[:300_000], V, n=16))
    tag = f"{cfg}_N{N}_n{n_eval}"
    np.save(os.path.join(HERE, "results", f"nll_{tag}.npy"), nll)
    peak = None
    if psutil:
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", mi.rss) / 2**20
    res = dict(cfg=cfg, n_train=N, n_eval=n_eval, data=DATA, bpc=float(nll.mean()),
               se_iid=float(nll.std(ddof=1) / np.sqrt(len(nll))), leak_free=leak,
               fit_wall_s=fit_wall, fit_cpu_s=fit_cpu, eval_wall_s=eval_wall, eval_chars_per_s=len(pos) / eval_wall,
               peak_rss_mb=peak, names=m.expert_names(), dev_bpc_experts=m.dev_bpc_experts,
               dev_bpc_mix=m.dev_bpc_mix, poe_w=None if m.poe_w is None else m.poe_w.tolist(),
               tau={e.name: float(e.params[0]) for e in m.xperts if e.mode == 1})
    json.dump(res, open(os.path.join(HERE, "results", f"lc_{tag}.json"), "w"), indent=1)
    print(f"[{cfg} N={N:,} n={n_eval}] bpc={res['bpc']:.4f} (SE iid {res['se_iid']:.4f}) leak_free={leak} "
          f"fit={fit_wall:.0f}s cpu={fit_cpu:.0f}s eval={eval_wall:.0f}s peak={peak} MB", flush=True)

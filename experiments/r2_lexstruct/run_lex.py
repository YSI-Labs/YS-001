"""Ablacion lexstruct sobre la base v0.1 (A+W+S+L). Uso: python -I run_lex.py <variante> [n_eval]
Variantes: base | <CADENA> (una cadena nueva de lex_chains) | all (todas) | <C1>+<C2>+... (combinacion).
Escribe results/lex_<variante>.json con bpc val (harness, n_eval=20000), dev_bpc por cadena, params, tiempos y pico de RAM."""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import ctypes
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
from core import Ixchel, default_chains, lex_chains


def peak_ws_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 1e6 if ok else -1.0


if __name__ == "__main__":
    var = sys.argv[1] if len(sys.argv) > 1 else "base"
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    allc = default_chains()
    lx = lex_chains()
    base = {k: allc[k] for k in "AWSL"}
    if var == "base":
        extra = []
    elif var == "all":
        extra = list(lx.keys())
    else:
        extra = var.split("+")
    chains = dict(base)
    for nm in extra:
        chains[nm] = lx[nm]
    m = Ixchel(chains=chains, gate_chains=("A", "W"), gated=True, name=f"Lex-{var}")
    res = harness.run(m, n_eval=n_eval, gen_chars=500)
    ntab = {nm: int(len(t["psym"])) for nm, t in m.tabs.items()}
    res.update(variant=var, added=extra, dev_bpc_chains=m.dev_bpc_chains, dev_bpc_mix=m.dev_bpc_mix,
               pairs_per_chain=ntab, peak_ws_mb=peak_ws_mb(), fit_model_s=m.fit_time)
    out = os.path.join(HERE, "results", f"lex_{var}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    harness.summarize(res)
    print("dev_bpc_mix", round(m.dev_bpc_mix, 4), "peak_MB", round(res["peak_ws_mb"], 0), flush=True)

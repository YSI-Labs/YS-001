"""Cuanto bpc pierde cuantizar los conteos de pares a codigos log de 8/7/6 bits (y colisiones de claves de 32 bits).
Ajusta AWSL con todo el train (core_opt, exacto), luego reemplaza pcnt por su version cuantizada y evalua en val
(n_eval=20000, como harness). Los descuentos D se mantienen (se calcularon con conteos exactos).
Uso: python -I compact_test.py"""
import os, sys, time, json
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import harness
import core_opt as co

train, val, itos = harness.load()
V = len(itos)
allc = co.default_chains()
m = co.Ixchel(chains={k: allc[k] for k in "AWSL"}, gate_chains=("A", "W"), gated=True, name="compact", verbose=False)
m.fit(train, V)
base_tabs = {nm: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in t.items()} for nm, t in m.tabs.items()}
res = {}


def qenc_dec(c, steps, K=16):
    """Codigo log: exacto hasta K; por encima, codigo = K + round(steps*log2(c/K)); decodifica a entero."""
    c = c.astype(np.float64)
    code = np.where(c <= K, c, K + np.rint(steps * np.log2(np.maximum(c, 1) / K)))
    dec = np.where(code <= K, code, np.rint(K * 2.0 ** ((code - K) / steps)))
    maxcode = int(code.max()) if code.size else 0
    return dec.astype(np.int32), maxcode


def eval_with(tabs, label):
    m.tabs = tabs
    r = harness.evaluate(m, val, V, n_eval=20000, check_leak=False)
    res[label] = r["bpc"]
    print(label, r["bpc"], flush=True)


eval_with(base_tabs, "exact_counts_bpc")
for steps in (4, 2, 1):
    tabs = {}
    maxcode = 0; nerr = 0; tot = 0
    for nm, t in base_tabs.items():
        t2 = dict(t)
        dec, mc = qenc_dec(t["pcnt"], steps)
        maxcode = max(maxcode, mc)
        nerr += int((dec != t["pcnt"]).sum()); tot += dec.size
        t2["pcnt"] = dec
        tabs[nm] = t2
    res[f"q_steps{steps}_max_code"] = maxcode
    res[f"q_steps{steps}_frac_pairs_changed"] = nerr / tot
    eval_with(tabs, f"q_log_steps{steps}_bpc")

# colisiones de clave de 32 bits (fingerprint = 32 bits bajos de la clave de contexto)
tot_ctx = 0; coll_ctx = 0
for nm, t in base_tabs.items():
    ck = t["ckey"]
    fp = ck & np.uint64(0xFFFFFFFF)
    u = np.unique(fp).size
    tot_ctx += ck.size; coll_ctx += ck.size - u
res["ctx_total"] = int(tot_ctx); res["ctx_colliding_fp32"] = int(coll_ctx)
res["ctx_colliding_fp32_frac"] = coll_ctx / max(tot_ctx, 1)
# tamano por par
pairs = sum(t["psym"].size for t in base_tabs.values())
ctxs = sum(t["ckey"].size for t in base_tabs.values())
res["pairs"] = int(pairs); res["contexts"] = int(ctxs)
res["bytes_per_pair_current"] = (sum(t["ckey"].nbytes + t["cstart"].nbytes + t["psym"].nbytes + t["pcnt"].nbytes
                                     for t in base_tabs.values())) / pairs
res["bytes_per_pair_compact_proposal"] = (pairs * 2 + ctxs * 8) / pairs   # sym7+cnt8 en 2 B; ctx: fp32 + offset32
res["ctx_per_pair"] = ctxs / pairs
json.dump(res, open(os.path.join(HERE, "results", "compact_test.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

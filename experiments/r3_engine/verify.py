"""Verificacion de equivalencia engine vs nucleo Ixchel.predict (copia en esta carpeta) + bpc del motor.
  A) 200 posiciones de val (harness.eval_positions): step() tras alimentar el prefijo vs predict(val,pos).
  B) generacion de 300 caracteres (prompt = val[1000:1064], seed 0): texto igual a harness.generate del nucleo y
     distribuciones del motor vs predict(seq, pos) del nucleo en cada posicion generada.
  C) bpc sobre n_eval=10000 posiciones de val: motor (streaming) vs nucleo (predict por lotes), SIN leak_check.
Salida: verify.json (cifras de aqui salen en el informe)."""
import os
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import harness
from core_v02 import Ixchel
from engine import IxchelEngine

MODEL = os.path.join(HERE, "model_v03")
INF = 1 << 40


def reference_from_pack(path):
    """Reconstruye un Ixchel del nucleo (codigo original sin tocar) con las mismas tablas empaquetadas."""
    meta = json.load(open(os.path.join(path, "meta.json")))
    ld = lambda k: np.load(os.path.join(path, k + ".npy"))
    G = {k: ld(k) for k in ("GCK", "GCS", "GPS", "GPC", "GCOFF", "GCSOFF", "GD", "LVL0", "NLVL", "CK_BASE",
                            "PS_BASE", "CS_BASE", "base", "poe_w")}
    m = Ixchel(name="Ixchel-v0.3-nucleo", chains=meta["chains"], cache_windows=tuple(meta["cache_windows"]),
               mix="poe", verbose=False)
    nch = len(meta["names"])
    tabs = {}
    for c, nm in enumerate(meta["names"]):
        ckb = int(G["CK_BASE"][c]); ckE = int(G["CK_BASE"][c + 1]) if c + 1 < nch else len(G["GCK"])
        psb = int(G["PS_BASE"][c]); psE = int(G["PS_BASE"][c + 1]) if c + 1 < nch else len(G["GPS"])
        csb = int(G["CS_BASE"][c]); csE = int(G["CS_BASE"][c + 1]) if c + 1 < nch else len(G["GCS"])
        lv = int(G["LVL0"][c]); L1 = int(G["NLVL"][c])
        tabs[nm] = dict(ckey=np.array(G["GCK"][ckb:ckE]), cstart=np.array(G["GCS"][csb:csE]) - psb,
                        psym=np.array(G["GPS"][psb:psE]), pcnt=np.array(G["GPC"][psb:psE]),
                        c_off=np.array(G["GCOFF"][lv:lv + L1 + 1]) - ckb,
                        cs_off=np.array(G["GCSOFF"][lv:lv + L1 + 1]) - csb,
                        D=np.array(G["GD"][lv:lv + L1]))
    m.tabs = tabs
    m.base = np.array(G["base"]); m.poe_w = np.array(G["poe_w"]); m.V = len(meta["itos"])
    m.train_index = None; m.hits = {}
    return m, meta


def bpc_from_probs(P, y):
    P = P.astype(np.float64)
    return float(-np.log2(np.maximum(P[np.arange(len(y)), y], 1e-12)).mean())


if __name__ == "__main__":
    train, val, itos = harness.load()
    V = len(itos)
    res = {}
    eng = IxchelEngine.load(MODEL)
    ref, meta = reference_from_pack(MODEL)
    res["model"] = dict(name=meta["name"], V=V, n_pairs=int(len(eng.A["GPS"])), model_bytes=eng.model_nbytes(),
                        fit_s=meta["fit_s"], dev_bpc_mix=meta["dev_bpc_mix"], poe_w=meta["poe_w"],
                        expert_names=eng.names + [f"C{w}" for w in meta["cache_windows"]])

    # ---------------- A) 200 posiciones de val
    pos = harness.eval_positions(val, 200)
    t0 = time.time(); Pref = ref.predict(val, pos); tref = time.time() - t0
    sess = eng.new_session()
    cur = 0; Peng = np.zeros_like(Pref)
    t0 = time.time()
    for i, p in enumerate(pos):
        eng.feed(sess, val[cur:p]); cur = int(p)
        Peng[i] = eng.step(sess)
    teng = time.time() - t0
    dA = np.abs(Peng.astype(np.float64) - Pref.astype(np.float64))
    res["A_val200"] = dict(n=int(len(pos)), max_abs_diff=float(dA.max()), mean_abs_diff=float(dA.mean()),
                           argmax_equal=float((Peng.argmax(1) == Pref.argmax(1)).mean()),
                           max_norm_err_engine=float(np.abs(Peng.sum(1) - 1).max()),
                           min_prob_engine=float(Peng.min()), bpc_ref_200=bpc_from_probs(Pref, val[pos]),
                           bpc_eng_200=bpc_from_probs(Peng, val[pos]), t_ref_batch_s=tref, t_eng_seq_s=teng,
                           passes_1e6=bool(dA.max() < 1e-6))
    print("A", res["A_val200"], flush=True)

    # ---------------- B) generacion 300 chars
    prompt = val[1000:1064]
    sess = eng.new_session(prompt)
    text_e, ids_e, D_e = eng.generate(sess, 300, seed=0, return_dists=True)
    text_r = harness.generate(ref, prompt, 300, V, seed=0, itos=itos)
    full = np.concatenate([prompt, ids_e]).astype(np.uint8)
    posB = np.arange(64, 364, dtype=np.int64)
    PrB = ref.predict(full, posB)
    dB = np.abs(D_e.astype(np.float64) - PrB.astype(np.float64))
    res["B_gen300"] = dict(text_equal_to_nucleo_generate=bool("".join(itos[i] for i in ids_e) == text_r[64:]),
                           chars_mismatch=int(sum(1 for a, b in zip(text_e, text_r[64:]) if a != b)),
                           max_abs_diff=float(dB.max()), mean_abs_diff=float(dB.mean()),
                           argmax_equal=float((D_e.argmax(1) == PrB.argmax(1)).mean()),
                           passes_1e6=bool(dB.max() < 1e-6), sample_head=text_e[:200])
    print("B", {k: v for k, v in res["B_gen300"].items() if k != "sample_head"}, flush=True)

    # ---------------- C) bpc n_eval=10000, motor en streaming vs nucleo por lotes
    n_eval = 10_000
    pos = harness.eval_positions(val, n_eval)
    t0 = time.time()
    sess = eng.new_session()
    cur = 0; nll = 0.0; worst = 0.0; minp = 1.0
    buf = np.zeros(V, np.float32)
    for p in pos:
        eng.feed(sess, val[cur:p]); cur = int(p)
        d = eng.step(sess)
        pt = float(d[val[p]])
        nll += -np.log2(max(pt, 1e-12))
        minp = min(minp, float(d.min()))
    t_stream = time.time() - t0
    bpc_eng = nll / len(pos)
    bres = harness.evaluate(ref, val, V, n_eval=n_eval, check_leak=False)
    res["C_bpc_10k"] = dict(n_eval=n_eval, bpc_engine=bpc_eng, bpc_nucleo=bres["bpc"],
                            abs_diff=abs(bpc_eng - bres["bpc"]), stream_s=t_stream, min_prob_engine=minp,
                            nucleo_max_norm_err=bres["max_norm_err"])
    print("C", res["C_bpc_10k"], flush=True)
    json.dump(res, open(os.path.join(HERE, "verify.json"), "w"), indent=1)
    print("OK verify.json")

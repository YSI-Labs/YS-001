"""Ixchel v0.3 como BASE del residuo. Dos modos:
  python -I base_fit.py full        -> tablas con todo el train; guarda LP de dev (tramo final 5%, tablas de train[:cut]),
                                       LP de val (10k, harness.eval_positions) y un chequeo de FUGA a proposito.
  python -I base_fit.py fold K      -> OOF: tablas con los otros 3 bloques de train[:cut]; LP de las primeras M posiciones
                                       del bloque K (contexto = solo el propio bloque, como val en inferencia).
"""
import os, sys, time, json
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import numpy as np
import harness
from ixchel_cfg import make, bpc_of_logp

M_SEG = 375_000        # posiciones por bloque (MVP: 4 x 375k = 1.5M OOF)
CTX_MIN = 64           # como harness.eval_positions (min_ctx=64)
K = 4

def chunked_logp(model, ids, pos, chunk=25_000):
    out = np.empty((len(pos), model.V), dtype=np.float32)
    for s in range(0, len(pos), chunk):
        P = model.predict(ids, pos[s:s + chunk])
        out[s:s + len(P)] = np.log(np.maximum(P.astype(np.float64), 1e-12)).astype(np.float32)
    return out

def blocks(train):
    n = len(train); cut = int(n * 0.95); B = cut // K
    return cut, B

if __name__ == "__main__":
    mode = sys.argv[1]
    train, val, itos = harness.load(); V = len(itos)
    cut, B = blocks(train)
    t0c, t0w = time.process_time(), time.time()
    if mode == "full":
        m = make("Ixchel-v0.3-full"); m.keep_dev = True
        m.fit(train, V)
        fit_cpu, fit_wall = time.process_time() - t0c, time.time() - t0w
        pos_v = harness.eval_positions(val, 10_000)
        LPv = chunked_logp(m, val, pos_v)
        bpc_val = bpc_of_logp(LPv, val[pos_v])
        # chequeo de FUGA a proposito: posiciones de train[:cut] predichas con tablas que YA contienen esos caracteres
        pos_leak = np.linspace(CTX_MIN, cut - 1, 20_000).astype(np.int64)
        LPleak = chunked_logp(m, train, pos_leak)
        bpc_leak = bpc_of_logp(LPleak, train[pos_leak])
        np.savez(os.path.join(HERE, "base_full.npz"), dev_logp=m.dev_logp, dev_pos=m.dev_pos_used,
                 LPv=LPv, pos_v=pos_v)
        res = dict(mode="full", fit_cpu_s=fit_cpu, fit_wall_s=fit_wall, bpc_val10k=bpc_val,
                   dev_bpc_poe=m.dev_bpc_mix, dev_positions=int(len(m.dev_pos_used)),
                   poe_w=m.poe_w.tolist(), expert_names=m.expert_names(),
                   bpc_train_leaky_20k=bpc_leak, n_params_pairs=m.n_params(), cut=cut, n=len(train))
        json.dump(res, open(os.path.join(HERE, "base_full.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
    elif mode == "fold":
        k = int(sys.argv[2])
        seg = train[k * B: k * B + M_SEG]
        other = np.concatenate([train[j * B:(j + 1) * B] for j in range(K) if j != k])
        m = make(f"Ixchel-v0.3-fold{k}")
        m.fit(other, V)
        fit_cpu, fit_wall = time.process_time() - t0c, time.time() - t0w
        pos = np.arange(CTX_MIN, M_SEG, dtype=np.int64)
        LP = chunked_logp(m, seg, pos)
        np.save(os.path.join(HERE, f"lp_fold{k}.npy"), LP.astype(np.float16))
        bpc = bpc_of_logp(LP, seg[pos])
        res = dict(mode="fold", k=k, n_pos=int(len(pos)), oof_bpc=bpc, fit_cpu_s=fit_cpu, fit_wall_s=fit_wall,
                   poe_w=m.poe_w.tolist(), dev_bpc_poe_inner=m.dev_bpc_mix,
                   block_start=k * B, block_len=B, fit_chars=int(len(other)))
        json.dump(res, open(os.path.join(HERE, f"base_fold{k}.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))

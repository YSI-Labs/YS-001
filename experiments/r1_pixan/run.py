"""Corrida final (val solo para reporte). Config elegida en dev: T=32, b=16, d=4, escalas 2^0..2^8.
Tambien corre la ablacion sin SimHash (ExactCtx k=4) y un diagnostico de buckets no vistos en dev."""
import sys, os; sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'experiments', 'r1_pixan'))
import numpy as np, devlib, model as M, harness
HERE = devlib.HERE
N_EVAL = 50_000

def unseen_rate():
    fit_part, dev_arr, V = devlib.dev_split()
    m = M.Pixan(T=32, b=16, d=4); m.fit(fit_part, V)
    pos = harness.eval_positions(dev_arr, 20000)
    out = {}
    for T in (1, 8, 32):
        N_all = []
        from numpy import asarray
        # recompute buckets for the eval positions
        import numpy as _np
        order = _np.argsort(pos); ps = pos[order]
        H = M.states_at(_np.ascontiguousarray(dev_arr), m.E2, m.a, ps); H -= m.mu; H /= m.sd
        B = m._buckets_from_std(H)
        for t in range(T):
            N_all.append(m.tables[t].tot[B[:, t]])
        N_all = _np.stack(N_all, 1)
        out[f"T{T}_frac_positions_with_zero_mass_in_all_tables"] = float((N_all.sum(1) == 0).mean())
        out[f"T{T}_frac_table_buckets_unseen"] = float((N_all == 0).mean())
    return out

if __name__ == "__main__":
    diag = unseen_rate()
    print("diag (dev):", diag, flush=True)
    pix = M.Pixan(T=32, b=16, d=4, name="Pixan-T32-b16-d4-K9")
    r1 = harness.run(pix, n_eval=N_EVAL, gen_chars=400, out_json=os.path.join(HERE, "result.json"),
                     extra=dict(dev_diag=diag))
    harness.summarize(r1)
    ex = M.ExactCtx(4)
    r2 = harness.run(ex, n_eval=N_EVAL, gen_chars=400, out_json=os.path.join(HERE, "result_exactctx4.json"))
    harness.summarize(r2)

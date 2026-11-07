"""Comparacion pareada posicion a posicion contra la base (mismas posiciones de val).
Uso: python -I compare.py <base_tag> <tag1> [tag2 ...]   (tags = cfg_N_n, p.ej. base_N3000000_n20000)
d = nll_base - nll_var  (d > 0 => la variante es MEJOR). SE iid y SE por bloques de 500 posiciones
(mismo estimador que ixchel/ensemble_test.py::block_stats)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def block_se(x, block=500):
    nb = len(x) // block
    bm = x[:nb * block].reshape(nb, block).mean(1)
    return float(bm.std(ddof=1) / np.sqrt(nb))


if __name__ == "__main__":
    base_tag = sys.argv[1]
    base = np.load(os.path.join(R, f"nll_{base_tag}.npy"))
    rows = []
    for tag in sys.argv[2:]:
        v = np.load(os.path.join(R, f"nll_{tag}.npy"))
        assert v.shape == base.shape, "posiciones distintas"
        d = base - v
        meta = json.load(open(os.path.join(R, f"lc_{tag}.json")))
        rows.append(dict(tag=tag, bpc=float(v.mean()), delta=float(d.mean()),
                         se_iid=float(d.std(ddof=1) / np.sqrt(len(d))), se_block=block_se(d),
                         frac_better=float((d > 0).mean()), fit_wall_s=meta["fit_wall_s"],
                         fit_cpu_s=meta["fit_cpu_s"], eval_chars_per_s=meta["eval_chars_per_s"],
                         leak_free=meta["leak_free"], tau=meta.get("tau")))
    bb = float(base.mean())
    print(f"base {base_tag}: bpc={bb:.4f}")
    print("| variante | bpc | delta vs base (bpc, + = mejor) | SE iid | SE bloques | frac. posiciones mejores | fit s | eval ch/s | leak | tau |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['tag']} | {r['bpc']:.4f} | {r['delta']:+.4f} | {r['se_iid']:.4f} | {r['se_block']:.4f} | "
              f"{r['frac_better']:.3f} | {r['fit_wall_s']:.0f} | {r['eval_chars_per_s']:.0f} | {r['leak_free']} | {r['tau']} |")
    json.dump(dict(base=base_tag, base_bpc=bb, rows=rows), open(os.path.join(R, "compare_" + "_".join(sys.argv[2:])[:80] + ".json"), "w"), indent=1)

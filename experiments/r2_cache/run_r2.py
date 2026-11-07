"""Corre variantes de Ixchel v0.2 (expertos de memoria en contexto) con el harness comun.
Uso: python -I run_r2.py <variante> [n_eval=20000]
Variantes: base (= v0.1, debe dar 2.078), cache, match, cache_match, cache_match_gx.
Escribe results/r2_<variante>.json (bpc val, params, tiempos, dev por experto, copia literal de 3000 chars)."""
import os, sys, json, time
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common')
sys.path.insert(0, HERE)
sys.path.insert(0, COMMON)
import numpy as np
import harness
from core import Ixchel
from cache_experts import WINDOWS_DEFAULT

ALL_WIN = tuple(WINDOWS_DEFAULT)
MATCH = ("hist", "train")
VARIANTS = {
    "base": dict(),                                                    # v0.1 reproducido
    "cache": dict(cache_windows=ALL_WIN),                              # + 4 expertos cache WB
    "match": dict(match_sources=MATCH),                                # + match hist + train
    "cache_match": dict(cache_windows=ALL_WIN, match_sources=MATCH),   # + ambos, gate v0.1
    "cache_match_gx": dict(cache_windows=ALL_WIN, match_sources=MATCH,
                           gate_extra=("cache_lvl", "match_b")),        # + ambos, gate extendida
}


def greedy_copy(text, train_str):
    """Mismo criterio que ROUND1 (r1_tzib/copy_stats.py): segmentacion voraz; desde cada posicion,
    la subcadena mas larga que aparece literalmente en el train (busqueda binaria, tope 2000)."""
    segs = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in train_str:
            segs.append(0)
            i += 1
            continue
        lo, hi = 1, min(n - i, 2000)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if text[i:i + mid] in train_str:
                lo = mid
            else:
                hi = mid - 1
        segs.append(lo)
        i += lo
    segs = np.array(segs)
    tot = segs.sum()
    return dict(n_segments=int(len(segs)),
                mean_seg_len=float(segs[segs > 0].mean()) if (segs > 0).any() else 0.0,
                char_weighted_mean=float((segs.astype(float) ** 2).sum() / max(tot, 1)),
                frac_in_seg_ge20=float(segs[segs >= 20].sum() / max(tot, 1)),
                frac_in_seg_ge50=float(segs[segs >= 50].sum() / max(tot, 1)),
                max_seg=int(segs.max()))


if __name__ == "__main__":
    var = sys.argv[1] if len(sys.argv) > 1 else "base"
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    kw = VARIANTS[var]
    m = Ixchel(name=f"Ixchel-v0.2-{var}", verbose=True, **kw)
    outp = os.path.join(HERE, "results", f"r2_{var}.json")
    t0 = time.time()
    res = harness.run(m, n_eval=n_eval, gen_chars=3000, out_json=None)
    # copia literal: 3000 chars generados (desde el caracter 64, como ROUND1)
    train, val, itos = harness.load()
    train_str = "".join(itos[i] for i in train)
    val_str = "".join(itos[i] for i in val[:6000])
    copy_gen = greedy_copy(res["sample"][64:], train_str)
    copy_val = greedy_copy(val_str, train_str)
    res.update(variant=var, n_eval=n_eval, gen_chars=3000,
               copy_generated=copy_gen, copy_val_reference=copy_val,
               dev_bpc_experts=m.dev_bpc_experts, dev_bpc_mix=m.dev_bpc_mix,
               match_hit_rates={k: v.tolist() for k, v in m.hits.items()},
               gate_cells=int(m.G), expert_names=m.expert_names(), wall_total_s=time.time() - t0,
               sample_head=res["sample"][:400])
    json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)
    harness.summarize(res)
    print("copia generada:", json.dumps(copy_gen), flush=True)
    print("copia val ref:", json.dumps(copy_val), flush=True)

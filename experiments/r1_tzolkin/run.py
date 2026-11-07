"""Corrida final reproducible (val, n_eval=50_000) de una configuracion de Tzolk'in.
Uso:  python -I run.py [nombre]      nombre por defecto = headline
La regla de combinacion (geo_seen) se eligio en dev_select.py con el 5% final de train.
"""
import os, sys, json
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import harness
from model import Tzolkin

CONFIGS = {
    # headline: 64 tablas, hash denso de 2^16 buckets, mascaras aleatorias, combinacion geo ponderada por 'seen'
    "headline": dict(M=64, b=16, mask="random", mix="geo_seen", hash="dense"),
    "M16_b16": dict(M=16, b=16, mask="random", mix="geo_seen", hash="dense"),
    "M16_b14": dict(M=16, b=14, mask="random", mix="geo_seen", hash="dense"),
    "M16_b18": dict(M=16, b=18, mask="random", mix="geo_seen", hash="dense"),
    "M16_exact": dict(M=16, b=0, mask="random", mix="geo_seen", hash="exact"),
    "M256_b14": dict(M=256, b=14, mask="random", mix="geo_seen", hash="dense"),
    "M16_contig_exact": dict(M=16, b=0, mask="contig", mix="geo_seen", hash="exact"),
    "M16_contig_b16": dict(M=16, b=16, mask="contig", mix="geo_seen", hash="dense"),
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "headline"
    cfg = CONFIGS[name]
    model = Tzolkin(**cfg, seed=0)
    model.name = f"Tzolkin[{name}]"
    out = os.path.join(HERE, "result.json" if name == "headline" else os.path.join("grid", f"{name}.json"))
    res = harness.run(model, n_eval=50_000, gen_chars=400, seed=0, extra=dict(config=cfg))
    res["n_nonzero"] = model.n_nonzero()  # despues del fit (no antes)
    res["n_params_alloc"] = int(model.M * (1 << model.b) * model.V) if model.hash == "dense" else res["n_nonzero"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    harness.summarize(res)
    print("n_nonzero", res["n_nonzero"], "n_params_alloc", res["n_params_alloc"])

"""Corrida final reproducible de Tz'ib (orden ilimitado sobre arreglo de sufijos).
Uso: python -I run.py [variante] [n_eval]   (por defecto WB-inf, 50000).
Escribe result_<variante>.json via harness.run(out_json=...). Escalado de datos: ver scale.py."""
import os, sys
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from model import Tzib
HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "WB-inf"
    n_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
    m = Tzib(variant)
    res = harness.run(m, n_eval=n_eval, gen_chars=600,
                      out_json=os.path.join(HERE, f"result_{variant}_{n_eval//1000}k.json"),
                      extra=dict(note="sa_size == train_chars"))
    harness.summarize(res)

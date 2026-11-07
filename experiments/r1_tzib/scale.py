import os, sys, json
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from model import Tzib
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for v in ["KN-5", "WB-inf", "KN-inf"]:
    for f in [0.25, 0.5, 1.0]:
        m = Tzib(v, train_frac=f)
        r = harness.run(m, n_eval=20000, gen_chars=100, extra=dict(note="sa_size == train_chars"))
        rows.append(dict(variant=v, frac=f, train_chars=int(r["train_chars"]), bpc=r["bpc"], n_params=r["n_params"]))
        print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(HERE, "scale.json"), "w"), indent=1)

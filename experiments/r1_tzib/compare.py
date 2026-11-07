import os, sys, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from model import Tzib, METHODS
HERE = os.path.dirname(os.path.abspath(__file__))
N_EVAL = int(os.environ.get("N_EVAL", "20000"))
OUT = os.path.join(HERE, f"out{N_EVAL//1000}k")
os.makedirs(OUT, exist_ok=True)
rows = []
for v in METHODS:
    m = Tzib(v)
    res = harness.run(m, n_eval=N_EVAL, gen_chars=400, out_json=os.path.join(OUT, f"{v}.json"))
    res["sa_size"] = m.sa_size; res["fit_cold"] = m.fit_info
    harness.summarize(res)
    print("", flush=True)
    rows.append(res)
print("| metodo | bpc | params | fit_wall_s | train ch/s | eval ch/s | leak_free |")
for r in rows:
    print(f"| {r['name']} | {r['bpc']:.4f} | {r['n_params']:,} | {r['fit_wall_s']:.2f} | {r['train_chars_per_s']:,.0f} | {r['eval_chars_per_s']:,.0f} | {r.get('leak_free')} |")

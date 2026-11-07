"""Diferencias pareadas de bpc vs base (mismas posiciones) con error estandar por bloques de 500."""
import sys, numpy as np
RES = "results"; BASE = "../../ixchel/results/mix15_base_nll.npz"
def pair(tag, base=BASE):
    a = np.load(base); b = np.load(f"{RES}/mix15_{tag}_nll.npz"); out = {}
    for d in ("web", "code", "math", "chat", "books"):
        x = a[d]; y = b[d]; n = min(len(x), len(y)); diff = (y[:n] - x[:n])
        nb = n // 500; bm = diff[:nb * 500].reshape(nb, 500).mean(1)
        out[d] = (float(diff.mean()), float(bm.std(ddof=1) / np.sqrt(nb)))
    return out
if __name__ == "__main__":
    for tag in sys.argv[1:]:
        r = pair(tag)
        print(tag, {d: f"{m:+.4f} ± {s:.4f}" for d, (m, s) in r.items()})

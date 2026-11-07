"""Medida de copia literal: segmentacion voraz. Desde cada posicion i, la subcadena mas larga
que aparece literalmente en el train. Se reporta la longitud media de esos segmentos (ponderada por
caracteres) y la fraccion de caracteres dentro de segmentos de longitud >= 20 y >= 50.
Referencia: el propio val (texto humano no visto en train) y una muestra de la salida del modelo."""
import os, sys, json, time
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, harness
from model import Tzib
HERE = os.path.dirname(os.path.abspath(__file__))

def greedy_copy(text, train_str, max_starts=None):
    segs = []; i = 0; n = len(text)
    while i < n:
        # busqueda: L maximo con text[i:i+L] in train (monotona)
        L = 1
        if text[i] not in train_str:
            segs.append(0); i += 1; continue
        lo, hi = 1, min(n - i, 2000)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if text[i:i + mid] in train_str: lo = mid
            else: hi = mid - 1
        segs.append(lo); i += lo
    segs = np.array(segs)
    tot = segs.sum()
    return dict(n_segments=int(len(segs)), mean_seg_len=float(segs[segs > 0].mean()) if (segs > 0).any() else 0.0,
                char_weighted_mean=float((segs.astype(float) ** 2).sum() / max(tot, 1)),
                frac_in_seg_ge20=float(segs[segs >= 20].sum() / max(tot, 1)),
                frac_in_seg_ge50=float(segs[segs >= 50].sum() / max(tot, 1)),
                max_seg=int(segs.max()))

if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "WB-inf"
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    train, val, itos = harness.load(); V = len(itos)
    train_str = "".join(itos[i] for i in train)
    m = Tzib(variant); m.fit(train, V)
    prompt = val[1000:1064]
    sample = harness.generate(m, prompt, n_gen, V, seed=0, itos=itos)
    val_str = "".join(itos[i] for i in val[:6000])
    res = dict(variant=variant, generated=greedy_copy(sample[64:], train_str), val_reference=greedy_copy(val_str, train_str),
               sample_head=sample[:400])
    print(json.dumps(res, indent=1, ensure_ascii=False))
    json.dump(res, open(os.path.join(HERE, f"copy_{variant}.json"), "w"), indent=1, ensure_ascii=False)

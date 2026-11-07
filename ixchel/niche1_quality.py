"""NICHO 1 (GO_TO_MARKET.md): puntuacion barata de calidad de datos con Ixchel v0.5 grande (big2) frente a un n-grama KN simple (cadena A, mismos datos)
y a la tasa de compresion zlib.
DISENO FIJADO ANTES DE VER RESULTADOS:
 - Docs: 500 documentos de C4 shard 00003 (NO usado por el entrenamiento: prep_big2 solo leyo 186k docs del shard 00001), >=1500 chars, primeros 1000 chars tras ASCII.
 - E1 degradacion controlada (limpio vs degradado, mismo doc): noise2, noise10 (sustituir 2%/10% de chars), wshuf (palabras barajadas), repeat (primeros 80 chars repetidos),
   splice (mitad de un doc + mitad de otro). Puntuacion = bits por caracter medios (posiciones 64..n-1); mayor = peor calidad. AUC no pareado (+ IC bootstrap por doc) y precision pareada.
 - E2 proxy natural: Spearman entre puntuacion y bpc de Qwen2.5-1.5B en 200 docs (script aparte niche1_qwen.py + niche1_analysis.py).
 - Regla de decision: Ixchel completo solo vale la pena frente a A si su AUC supera al de A con IC95% no solapado en la mayoria de degradaciones Y supera a zlib.
Uso: python -I niche1_quality.py"""
import os, sys, json, time, zlib, gzip
os.environ["IXCHEL_DATA"] = "big2"; os.environ["NUMBA_NUM_THREADS"] = "8"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, eval_suite as ES
from core_v05 import Ixchel
from chains_v05 import v05_chains
from longcache import KeyedWin
from bigbuild import load_tables
from mixing import poe_combine
INF = 1 << 40
N_DOCS, DOC_CHARS = 500, 1000
SEED = 20261106

def load_docs(itos, enc):
    docs = []
    with gzip.open(os.path.join(ROOT, "data", "raw_mix", "c4big", "c4_00003.json.gz"), "rt", encoding="utf-8") as f:
        for ln in f:
            try: t = json.loads(ln)["text"]
            except Exception: continue
            ids = enc(t)
            if len(ids) < 1500: continue
            docs.append("".join(itos[i] for i in ids[:DOC_CHARS]))
            if len(docs) >= N_DOCS: break
    return docs

def degrade(docs, rng):
    letters = list("abcdefghijklmnopqrstuvwxyz")
    out = {k: [] for k in ("noise2", "noise10", "wshuf", "repeat", "splice")}
    for i, d in enumerate(docs):
        for k, fr in (("noise2", 0.02), ("noise10", 0.10)):
            a = list(d); m = rng.random(len(a)) < fr
            for j in np.nonzero(m)[0]: a[j] = letters[rng.integers(26)]
            out[k].append("".join(a))
        w = d.split(" "); rng.shuffle(w); s = " ".join(w); out["wshuf"].append((s + " " * DOC_CHARS)[:DOC_CHARS])
        r = d[:80]; out["repeat"].append((r * (DOC_CHARS // len(r) + 1))[:DOC_CHARS])
        o = docs[(i + 1 + int(rng.integers(len(docs) - 1))) % len(docs)]; out["splice"].append(d[:DOC_CHARS // 2] + o[DOC_CHARS // 2:DOC_CHARS])
    return out

def main():
    D = os.path.join(ROOT, "data", "big2")
    train = np.load(os.path.join(D, "train.npy"), mmap_mode="r"); man = json.load(open(os.path.join(D, "manifest.json"))); cut = int(man["main_chars_with_repetition"]); n = len(train)
    itos = json.load(open(os.path.join(D, "vocab.json"))); V = len(itos); enc = ES.make_encoder(itos)
    m = Ixchel(name="Ixchel-v05-big2-niche1", verbose=False, chains=v05_chains(), mix="poe", cache_windows=(), dev_frac=1 - cut / n,
               xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))
    m.dev_span = 10_000_000; m.dev_positions = 150_000
    m.prebuilt_tabs = load_tables(os.path.join(ROOT, "data", "big2_model"), m.names, mmap=True)
    cnt = np.zeros(V)
    for i in range(0, cut, 50_000_000): cnt += np.bincount(np.asarray(train[i:min(i + 50_000_000, cut)]), minlength=V)
    m.base_override = cnt / cnt.sum()
    t = time.time(); m.fit(train, V); print("fit", round(time.time() - t, 1), "s; expertos:", m.expert_names(), flush=True)
    assert m.names[0] == "A"

    def score(text):
        ids = enc(text).astype(np.uint8); pos = np.arange(64, len(ids), dtype=np.int64)
        A = m._atoms_for(ids); ctx = m._ctx_prepare(ids, int(pos.max()), m.base, m.train_index)
        for e in m.xperts: e.prepare(ids)
        Ps, _ = m._expert_dists(m.tabs, A, pos, ids, pos, m.V, ctx)
        full = poe_combine([np.log(np.maximum(p, 1e-12)) for p in Ps], m.poe_w).astype(np.float64)
        tg = ids[pos]; r = np.arange(len(pos))
        nf = -np.log2(np.maximum(full[r, tg], 1e-12)).mean(); na = -np.log2(np.maximum(Ps[0][r, tg].astype(np.float64), 1e-12)).mean()
        return nf, na, len(zlib.compress(text.encode("ascii", "replace"), 9)) / len(text)

    docs = load_docs(itos, enc); print("docs", len(docs), flush=True)
    rng = np.random.default_rng(SEED); variants = {"clean": docs, **degrade(docs, rng)}
    json.dump(dict(docs=docs[:200], seed=SEED), open(os.path.join(HERE, "results", "niche1_docs.json"), "w"))
    res = {}; t = time.time(); nchar = 0
    for k, lst in variants.items():
        sc = np.array([score(x) for x in lst]); res[k] = sc; nchar += sum(len(x) for x in lst)
        print(k, "full", round(sc[:, 0].mean(), 3), "A", round(sc[:, 1].mean(), 3), "zlib", round(sc[:, 2].mean(), 3), f"({time.time() - t:.0f}s)", flush=True)
    np.savez(os.path.join(HERE, "results", "niche1_scores.npz"), **res)
    print("NICHE1_DONE; chars/s", round(nchar / (time.time() - t)), flush=True)

if __name__ == "__main__":
    main()

"""PRUEBA DE ENTRENAR CON / SIN FILTRO (nicho 1, GO_TO_MARKET.md): prepara el conjunto y las puntuaciones.
Docs de C4 shard 00003 (no usado por el entrenamiento de Ixchel): se saltan los 500 primeros docs validos (usados en niche1_*), luego 8000 docs de POOL y 400 de EVAL,
1000 caracteres cada uno. 'Contaminado': 30% de los docs del pool (elegidos al azar, semilla fija) se sustituyen por una version degradada (tipo al azar entre ruido 10%,
palabras barajadas, repeticion de 80 caracteres, empalme con otro doc). Se puntua con Ixchel v0.5 completo (con cache) y con la cadena A (KN). Guarda data/filter_exp/pool.npz.
Uso: python -I filter_prepare.py"""
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
INF = 1 << 40; SKIP, NPOOL, NEVAL, L = 500, 8000, 400, 1000; SEED = 20261107; JUNK_FRAC = 0.30

def main():
    D = os.path.join(ROOT, "data", "big2"); OUT = os.path.join(ROOT, "data", "filter_exp"); os.makedirs(OUT, exist_ok=True)
    train = np.load(os.path.join(D, "train.npy"), mmap_mode="r"); man = json.load(open(os.path.join(D, "manifest.json"))); cut = int(man["main_chars_with_repetition"]); n = len(train)
    itos = json.load(open(os.path.join(D, "vocab.json"))); V = len(itos); enc = ES.make_encoder(itos)
    docs = []
    with gzip.open(os.path.join(ROOT, "data", "raw_mix", "c4big", "c4_00003.json.gz"), "rt", encoding="utf-8") as f:
        for ln in f:
            try: t = json.loads(ln)["text"]
            except Exception: continue
            ids = enc(t)
            if len(ids) < 1500: continue
            docs.append("".join(itos[i] for i in ids[:L]))
            if len(docs) >= SKIP + NPOOL + NEVAL: break
    pool, evald = docs[SKIP:SKIP + NPOOL], docs[SKIP + NPOOL:]
    rng = np.random.default_rng(SEED); letters = list("abcdefghijklmnopqrstuvwxyz")
    junk_idx = np.sort(rng.choice(NPOOL, int(JUNK_FRAC * NPOOL), replace=False)); junk_docs = []; junk_type = []
    for i in junk_idx:
        d = pool[i]; ty = int(rng.integers(4)); junk_type.append(ty)
        if ty == 0:
            a = list(d); mk = rng.random(len(a)) < 0.10
            for j in np.nonzero(mk)[0]: a[j] = letters[rng.integers(26)]
            s = "".join(a)
        elif ty == 1:
            w = d.split(" "); rng.shuffle(w); s = " ".join(w)
        elif ty == 2:
            r = d[:80]; s = r * (L // 80 + 1)
        else:
            o = pool[int(rng.integers(NPOOL))]; s = d[:L // 2] + o[L // 2:]
        junk_docs.append((s + " " * L)[:L])
    m = Ixchel(name="Ixchel-v05-big2-filter", verbose=False, chains=v05_chains(), mix="poe", cache_windows=(), dev_frac=1 - cut / n,
               xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))
    m.dev_span = 10_000_000; m.dev_positions = 150_000
    m.prebuilt_tabs = load_tables(os.path.join(ROOT, "data", "big2_model"), m.names, mmap=True)
    cnt = np.zeros(V)
    for i in range(0, cut, 50_000_000): cnt += np.bincount(np.asarray(train[i:min(i + 50_000_000, cut)]), minlength=V)
    m.base_override = cnt / cnt.sum(); t = time.time(); m.fit(train, V); print("fit", round(time.time() - t, 1), flush=True)

    def score(text):
        ids = enc(text).astype(np.uint8); pos = np.arange(64, len(ids), dtype=np.int64)
        A = m._atoms_for(ids); ctx = m._ctx_prepare(ids, int(pos.max()), m.base, m.train_index)
        for e in m.xperts: e.prepare(ids)
        Ps, _ = m._expert_dists(m.tabs, A, pos, ids, pos, m.V, ctx)
        full = poe_combine([np.log(np.maximum(p, 1e-12)) for p in Ps], m.poe_w).astype(np.float64)
        tg = ids[pos]; r = np.arange(len(pos))
        return -np.log2(np.maximum(full[r, tg], 1e-12)).mean(), -np.log2(np.maximum(Ps[0][r, tg].astype(np.float64), 1e-12)).mean(), len(zlib.compress(text.encode("ascii", "replace"), 9)) / len(text)
    t = time.time(); sc_pool = np.array([score(x) for x in pool]); print("pool puntuado", round(time.time() - t), "s", flush=True)
    t = time.time(); sc_junk = np.array([score(x) for x in junk_docs]); print("junk puntuado", round(time.time() - t), "s", flush=True)
    E = lambda lst: np.stack([enc(x).astype(np.uint8)[:L] for x in lst])
    np.savez(os.path.join(OUT, "pool.npz"), pool=E(pool), junk=E(junk_docs), junk_idx=junk_idx, junk_type=np.array(junk_type), eval=E(evald), sc_pool=sc_pool, sc_junk=sc_junk, V=V)
    print("medias limpio [full, A, zlib]:", sc_pool.mean(0).round(3), "junk:", sc_junk.mean(0).round(3), "FILTER_PREP_DONE", flush=True)

if __name__ == "__main__":
    main()

"""NICHO 1, seguimiento (a): Ixchel SIN cache de historial (expertos de conteo puros, pesos PoE reajustados) sobre los MISMOS documentos y degradaciones
que niche1_quality.py (misma semilla), mas un detector de repeticion = (bpc sin cache) - (bpc con cache). Guarda results/niche1_nocache_scores.npz.
DISENO FIJADO ANTES DE VER RESULTADOS:
 - 'nocache' se considera corregido si su AUC en 'repeat' > 0.5 con IC95% por documento por encima de 0.5; el detector de repeticion (gap) se evalua con AUC en 'repeat' vs 'clean'.
 - Puntuador combinado SIN constantes libres: rango-maximo de {bpc sin cache, gap} (un doc se marca de baja calidad si es alto en cualquiera de los dos rangos entre los docs evaluados).
Uso: python -I niche1_nocache.py"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "big2"; os.environ["NUMBA_NUM_THREADS"] = "8"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, eval_suite as ES
from core_v05 import Ixchel
from chains_v05 import v05_chains
from bigbuild import load_tables
from mixing import poe_combine
import niche1_quality as Q

def main():
    D = os.path.join(ROOT, "data", "big2")
    train = np.load(os.path.join(D, "train.npy"), mmap_mode="r"); man = json.load(open(os.path.join(D, "manifest.json"))); cut = int(man["main_chars_with_repetition"]); n = len(train)
    itos = json.load(open(os.path.join(D, "vocab.json"))); V = len(itos); enc = ES.make_encoder(itos)
    m = Ixchel(name="Ixchel-v05-big2-nocache", verbose=False, chains=v05_chains(), mix="poe", cache_windows=(), dev_frac=1 - cut / n, xperts=())
    m.dev_span = 10_000_000; m.dev_positions = 150_000
    m.prebuilt_tabs = load_tables(os.path.join(ROOT, "data", "big2_model"), m.names, mmap=True)
    cnt = np.zeros(V)
    for i in range(0, cut, 50_000_000): cnt += np.bincount(np.asarray(train[i:min(i + 50_000_000, cut)]), minlength=V)
    m.base_override = cnt / cnt.sum()
    t = time.time(); m.fit(train, V); print("fit", round(time.time() - t, 1), "s; expertos:", m.expert_names(), "dev bpc", m.dev_bpc_mix, flush=True)
    assert m.names[0] == "A" and len(m.expert_names()) == 14

    def score(text):
        ids = enc(text).astype(np.uint8); pos = np.arange(64, len(ids), dtype=np.int64)
        A = m._atoms_for(ids); ctx = m._ctx_prepare(ids, int(pos.max()), m.base, m.train_index)
        Ps, _ = m._expert_dists(m.tabs, A, pos, ids, pos, m.V, ctx)
        full = poe_combine([np.log(np.maximum(p, 1e-12)) for p in Ps], m.poe_w).astype(np.float64)
        tg = ids[pos]; r = np.arange(len(pos))
        return -np.log2(np.maximum(full[r, tg], 1e-12)).mean(), -np.log2(np.maximum(Ps[0][r, tg].astype(np.float64), 1e-12)).mean()

    docs = Q.load_docs(itos, enc); rng = np.random.default_rng(Q.SEED); variants = {"clean": docs, **Q.degrade(docs, rng)}
    res = {}; t = time.time()
    for k, lst in variants.items():
        res[k] = np.array([score(x) for x in lst]); print(k, "nocache", round(res[k][:, 0].mean(), 3), "A", round(res[k][:, 1].mean(), 3), f"({time.time() - t:.0f}s)", flush=True)
    np.savez(os.path.join(HERE, "results", "niche1_nocache_scores.npz"), **res)
    old = np.load(os.path.join(HERE, "results", "niche1_scores.npz"))
    print("control A identica a la corrida anterior:", float(np.abs(old["clean"][:, 1] - res["clean"][:, 1]).max()), flush=True)
    print("NOCACHE_DONE", flush=True)

if __name__ == "__main__":
    main()

"""NICHO 4 (GO_TO_MARKET.md): compresion de texto/codigo. Longitud de codigo IDEAL (cota de Shannon: -log2 p de cada caracter, codificacion secuencial causal de
todo el archivo de validacion) de Ixchel v0.5 grande frente a compresores estandar sobre los MISMOS archivos (val_* de big2, decodificados a ASCII).
NO se implementa el codificador aritmetico: el tamano de Ixchel es la cota teorica (un codificador real anade una sobrecarga pequena [M], no medida aqui).
Compresores: gzip(zlib 9), bzip2 9, xz 9e, brotli q11, zstd 22 (con --long), zstd 19 con DICCIONARIO entrenado en datos del TRAIN (112 KB y 1 MB).
Contabilidad: se informa el tamano del archivo comprimido Y el punto de equilibrio si se cuenta el tamano del modelo (16.9 GB) como parte del descompresor.
Los primeros 64 caracteres se cuentan con la distribucion de orden 0 del train. Uso: python -I niche4_compress.py"""
import os, sys, json, time, zlib, bz2, lzma
os.environ["IXCHEL_DATA"] = "big2"; os.environ["NUMBA_NUM_THREADS"] = "8"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, zstandard, brotli, eval_suite as ES
from core_v05 import Ixchel
from chains_v05 import v05_chains
from longcache import KeyedWin
from bigbuild import load_tables
INF = 1 << 40; MODEL_BYTES = 16.93 * 2**30

def main():
    D = os.path.join(ROOT, "data", "big2")
    train = np.load(os.path.join(D, "train.npy"), mmap_mode="r"); man = json.load(open(os.path.join(D, "manifest.json"))); cut = int(man["main_chars_with_repetition"]); n = len(train)
    itos = json.load(open(os.path.join(D, "vocab.json"))); V = len(itos)
    # --- diccionarios zstd entrenados con muestras del train (ventanas de 16 KB de la parte principal)
    rng = np.random.default_rng(0); samples = []
    for st in rng.integers(0, cut - 20000, 4000): samples.append("".join(itos[i] for i in np.asarray(train[st:st + 16384])).encode("ascii", "replace"))
    dicts = {"zstd19+dict112KB": zstandard.train_dictionary(112 * 1024, samples), "zstd19+dict1MB": zstandard.train_dictionary(1024 * 1024, samples)}
    print("diccionarios entrenados", {k: len(v.as_bytes()) for k, v in dicts.items()}, flush=True)
    m = Ixchel(name="Ixchel-v05-big2-niche4", verbose=False, chains=v05_chains(), mix="poe", cache_windows=(), dev_frac=1 - cut / n,
               xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))
    m.dev_span = 10_000_000; m.dev_positions = 150_000
    m.prebuilt_tabs = load_tables(os.path.join(ROOT, "data", "big2_model"), m.names, mmap=True)
    cnt = np.zeros(V)
    for i in range(0, cut, 50_000_000): cnt += np.bincount(np.asarray(train[i:min(i + 50_000_000, cut)]), minlength=V)
    base = cnt / cnt.sum(); m.base_override = base
    t = time.time(); m.fit(train, V); print("fit", round(time.time() - t, 1), "s", flush=True)
    res = {}
    for d in ("web", "code", "math", "chat", "books"):
        val = np.load(os.path.join(D, f"val_{d}.npy")); N = len(val); text = "".join(itos[i] for i in val).encode("ascii", "replace")
        t = time.time(); bits = float(-np.log2(base[np.asarray(val[:64])]).sum()); pos_all = np.arange(64, N, dtype=np.int64)
        for s in range(0, len(pos_all), 8192):
            pp = pos_all[s:s + 8192]; P = m.predict(val, pp).astype(np.float64)
            bits += float(-np.log2(np.maximum(P[np.arange(len(pp)), val[pp]], 1e-12)).sum())
        t_ix = time.time() - t; r = dict(chars=N, ixchel_bytes=bits / 8, ixchel_bpc=bits / N, ixchel_chars_per_s=N / t_ix, raw_bytes=len(text))
        t = time.time()
        r["gzip9"] = len(zlib.compress(text, 9)); r["bzip2_9"] = len(bz2.compress(text, 9)); r["xz9e"] = len(lzma.compress(text, preset=9 | lzma.PRESET_EXTREME))
        r["brotli11"] = len(brotli.compress(text, quality=11, lgwin=24))
        r["zstd22"] = len(zstandard.ZstdCompressor(level=22).compress(text))
        for k, dd in dicts.items(): r[k] = len(zstandard.ZstdCompressor(level=19, dict_data=dd).compress(text))
        r["std_compress_s"] = time.time() - t; res[d] = r
        print(d, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        json.dump(dict(per_domain=res, model_bytes=MODEL_BYTES), open(os.path.join(HERE, "results", "niche4_compress.json"), "w"), indent=1)
    print("NICHE4_DONE", flush=True)

if __name__ == "__main__":
    main()

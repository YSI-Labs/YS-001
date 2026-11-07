"""NICHO 4, seguimiento: compresor REAL (codificador aritmetico entero de 32 bits, estilo Witten-Neal-Cleary, frecuencias cuantizadas a total <= 65536) con Ixchel v0.5 como modelo,
con prueba de IDA Y VUELTA (decodificar y comparar byte a byte). Codificador y decodificador usan la MISMA ruta de calculo (predict de una posicion con historial = prefijo + 1 marcador),
dentro del mismo proceso: el determinismo ENTRE MAQUINAS/hilos no se prueba aqui. Los primeros 64 caracteres usan la distribucion de orden 0 del train.
Extractos: primeros N caracteres de val_web y val_code (N=20000 por defecto). Uso: python -I niche4_coder.py [N]"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "big2"; os.environ["NUMBA_NUM_THREADS"] = "8"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np
from core_v05 import Ixchel
from chains_v05 import v05_chains
from longcache import KeyedWin
from bigbuild import load_tables
INF = 1 << 40; TOT = 65536; HALF = 1 << 31; QUART = 1 << 30; TOP = (1 << 32) - 1

def quant(p, V):
    p = np.asarray(p, dtype=np.float64); p = p / p.sum(); f = 1 + np.floor(p * (TOT - V)).astype(np.int64); return f

class Enc:
    def __init__(self): self.low, self.high, self.pend, self.bits = 0, TOP, 0, []
    def _out(self, b):
        self.bits.append(b)
        if self.pend: self.bits.extend([1 - b] * self.pend); self.pend = 0
    def put(self, clo, chi, tot):
        r = self.high - self.low + 1; self.high = self.low + r * chi // tot - 1; self.low = self.low + r * clo // tot
        while True:
            if self.high < HALF: self._out(0)
            elif self.low >= HALF: self._out(1); self.low -= HALF; self.high -= HALF
            elif self.low >= QUART and self.high < 3 * QUART: self.pend += 1; self.low -= QUART; self.high -= QUART
            else: break
            self.low = self.low * 2; self.high = self.high * 2 + 1
    def finish(self):
        self.pend += 1; self._out(0 if self.low < QUART else 1)
        b = self.bits + [0] * ((-len(self.bits)) % 8); return bytes(int("".join(map(str, b[i:i + 8])), 2) for i in range(0, len(b), 8))

class Dec:
    def __init__(self, data):
        self.bits = "".join(f"{x:08b}" for x in data); self.pos = 0; self.low, self.high = 0, TOP; self.val = 0
        for _ in range(32): self.val = self.val * 2 + self._bit()
    def _bit(self):
        b = int(self.bits[self.pos]) if self.pos < len(self.bits) else 0; self.pos += 1; return b
    def get(self, cum, tot):
        r = self.high - self.low + 1; target = ((self.val - self.low + 1) * tot - 1) // r
        s = int(np.searchsorted(cum, target, side="right")) - 1; clo, chi = int(cum[s]), int(cum[s + 1])
        self.high = self.low + r * chi // tot - 1; self.low = self.low + r * clo // tot
        while True:
            if self.high < HALF: pass
            elif self.low >= HALF: self.low -= HALF; self.high -= HALF; self.val -= HALF
            elif self.low >= QUART and self.high < 3 * QUART: self.low -= QUART; self.high -= QUART; self.val -= QUART
            else: break
            self.low = self.low * 2; self.high = self.high * 2 + 1; self.val = self.val * 2 + self._bit()
        return s

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    D = os.path.join(ROOT, "data", "big2")
    train = np.load(os.path.join(D, "train.npy"), mmap_mode="r"); man = json.load(open(os.path.join(D, "manifest.json"))); cut = int(man["main_chars_with_repetition"]); n = len(train)
    itos = json.load(open(os.path.join(D, "vocab.json"))); V = len(itos)
    m = Ixchel(name="Ixchel-v05-big2-coder", verbose=False, chains=v05_chains(), mix="poe", cache_windows=(), dev_frac=1 - cut / n,
               xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))
    m.dev_span = 10_000_000; m.dev_positions = 150_000
    m.prebuilt_tabs = load_tables(os.path.join(ROOT, "data", "big2_model"), m.names, mmap=True)
    cnt = np.zeros(V)
    for i in range(0, cut, 50_000_000): cnt += np.bincount(np.asarray(train[i:min(i + 50_000_000, cut)]), minlength=V)
    base = cnt / cnt.sum(); m.base_override = base; t = time.time(); m.fit(train, V); print("fit", round(time.time() - t, 1), flush=True)
    fb = quant(base, V); cumb = np.concatenate([[0], np.cumsum(fb)])

    def probs(hist, p):          # hist: ids[:p] decodificados; devuelve P para la posicion p
        arr = np.concatenate([hist[:p], np.zeros(1, dtype=np.uint8)]).astype(np.uint8)
        return m.predict(arr, np.array([p], dtype=np.int64))[0]
    res = {}
    import zlib, lzma, bz2, brotli, zstandard, pyppmd
    for name in ("web", "code"):
        val = np.load(os.path.join(D, f"val_{name}.npy"))[:N].astype(np.uint8); ideal = 0.0; q_ideal = 0.0
        t = time.time(); e = Enc()
        for p in range(len(val)):
            s = int(val[p])
            if p < 64: f, cum = fb, cumb; pr = base
            else: pr = probs(val, p); f = quant(pr, V); cum = np.concatenate([[0], np.cumsum(f)])
            tot = int(cum[-1]); e.put(int(cum[s]), int(cum[s + 1]), tot)
            ideal += -np.log2(max(float(pr[s]) / float(np.sum(pr)), 1e-12)); q_ideal += -np.log2(f[s] / tot)
        data = e.finish(); t_enc = time.time() - t
        t = time.time(); d = Dec(data); out = np.zeros(len(val), dtype=np.uint8)
        for p in range(len(val)):
            if p < 64: s = d.get(cumb, int(cumb[-1]))
            else: pr = probs(out, p); f = quant(pr, V); cum = np.concatenate([[0], np.cumsum(f)]); s = d.get(cum, int(cum[-1]))
            out[p] = s
        t_dec = time.time() - t; ok = bool(np.array_equal(out, val))
        text = "".join(itos[i] for i in val).encode("ascii", "replace")
        std = dict(gzip9=len(zlib.compress(text, 9)), bzip2_9=len(bz2.compress(text, 9)), xz9e=len(lzma.compress(text, preset=9 | lzma.PRESET_EXTREME)), brotli11=len(brotli.compress(text, quality=11)),
                   zstd22=len(zstandard.ZstdCompressor(level=22).compress(text)), ppmd_o8=len(pyppmd.compress(text, max_order=8, mem_size=64 << 20)), ppmd_o16=len(pyppmd.compress(text, max_order=16, mem_size=64 << 20)))
        r = dict(chars=int(len(val)), roundtrip_ok=ok, coded_bytes=len(data), ideal_bytes=ideal / 8, quantized_ideal_bytes=q_ideal / 8, overhead_vs_ideal=len(data) / (ideal / 8) - 1,
                 coded_bpc=len(data) * 8 / len(val), encode_s=t_enc, decode_s=t_dec, encode_chars_per_s=len(val) / t_enc, decode_chars_per_s=len(val) / t_dec, standard_on_same_extract=std)
        res[name] = r; print(name, json.dumps(r), flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "niche4_coder.json"), "w"), indent=1)
    print("NICHE4_CODER_DONE", flush=True)

if __name__ == "__main__":
    main()

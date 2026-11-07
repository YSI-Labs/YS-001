"""Tzolk'in: ensamble de tablas de conteo indexadas por mascaras aleatorias de contexto.

Cada tabla m tiene una mascara de offsets O_m (subconjunto de {1..L}, siempre incluye el offset 1,
el resto se elige sin reemplazo con peso 1/offset). El contexto de una posicion es la tupla
(ids[t-o] for o in O_m), codificada en un entero base V+1 (V = centinela de inicio) y hasheada a
2^b buckets. Entrenamiento = un conteo: T_m[bucket, siguiente] += 1 (sin gradientes, sin LR).

Prediccion por tabla (parameter-free, estilo Witten-Bell con base unigrama):
    p_m(v) = (c_v + U_m * q(v)) / (T_m + U_m),   T_m = total, U_m = tipos (celdas > 0), q = unigrama
    w_m    = T_m / (T_m + U_m)                    (evidencia: 1 - probabilidad de escape)
Combinacion (sin parametros libres):
    'lin_wb' : p = sum_m w_m p_m / sum_m w_m              (mezcla lineal ponderada por evidencia)
    'geo_wb' : p ~ exp( sum_m w_m log p_m / sum_m w_m )   (producto geometrico ponderado, normalizado)
    'lin_T'  : p = sum_m T_m p_m / sum_m T_m              (ponderada por conteo bruto)
Si un bucket nunca se vio (T=0) la tabla no aporta (w=0). Si ninguna tabla aporta -> q.

Constantes libres (ver REPORT.md, auditoria): M, L, kmax, b, semilla, prior de mascara 1/o,
el offset 1 siempre incluido, el constante de hash (Fibonacci), tope de contador (uint16 con
desplazamiento de escala), y la regla de combinacion.
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
import numpy as np

HASH_C = np.uint64(0x9E3779B97F4A7C15)  # multiplicador de Fibonacci (constante fija, no tuneada)


class Tzolkin:
    def __init__(self, M=16, L=16, kmax=8, b=16, mask="random", mix="lin_wb", seed=0, hash="dense"):
        self.M, self.L, self.kmax, self.b = M, L, kmax, b
        self.mask, self.mix, self.seed, self.hash = mask, mix, seed, hash
        self.name = f"Tzolkin-M{M}-{'exact' if hash == 'exact' else 'b%d' % b}-{mask}-{mix}"

    # ---------- mascaras ----------
    def _make_masks(self):
        rng = np.random.default_rng(self.seed)
        self.masks = []
        others = np.arange(2, self.L + 1)
        w = 1.0 / others
        w = w / w.sum()
        for _ in range(self.M):
            k = int(rng.integers(1, self.kmax + 1))  # k uniforme en 1..kmax
            if self.mask == "contig":
                offs = np.arange(1, k + 1)
            else:
                rest = rng.choice(others, size=k - 1, replace=False, p=w) if k > 1 else np.array([], dtype=np.int64)
                offs = np.sort(np.concatenate([[1], rest])).astype(np.int64)
            self.masks.append(offs)

    # ---------- hash ----------
    def _keys(self, xp, t, offs):
        key = np.zeros(len(t), dtype=np.int64)
        mul = 1
        base = self.V + 1
        for o in offs:
            key += xp[t - o + self.L].astype(np.int64) * mul
            mul *= base
        return key

    def _buckets(self, key):
        h = (key.astype(np.uint64) + np.uint64(1)) * HASH_C  # wrap-around uint64 es intencionado
        return (h >> np.uint64(64 - self.b)).astype(np.int64)

    # ---------- contrato ----------
    def fit(self, train, V):
        self.V = V
        n = len(train)
        self._make_masks()
        self.uni = np.bincount(train, minlength=V).astype(np.float64)
        U0 = float((self.uni > 0).sum())
        self.q = (self.uni + U0 / V) / (n + U0)  # unigrama con suavizado WB (como baseline_wb)
        xp = np.concatenate([np.full(self.L, V, np.uint8), train])
        t = np.arange(n, dtype=np.int64)
        tgt = train.astype(np.int64)
        self.tabs, self.shifts, self.ex = [], [], []
        if self.hash == "exact":  # sin colisiones: pares (contexto, siguiente) ordenados + totales por contexto
            for offs in self.masks:
                ck = self._keys(xp, t, offs)
                pk, pc = np.unique(ck * V + tgt, return_counts=True)
                ctx = pk // V
                cu, idx, typ = np.unique(ctx, return_index=True, return_counts=True)
                ctot = np.add.reduceat(pc, idx).astype(np.float64)
                self.ex.append((cu, ctot, typ.astype(np.float64), pk, pc.astype(np.float64)))
            return
        nb = 1 << self.b
        for offs in self.masks:
            bk = self._buckets(self._keys(xp, t, offs))
            cnt = np.bincount(bk * V + tgt, minlength=nb * V).reshape(nb, V)
            mx = int(cnt.max()); s = 0
            while (mx >> s) > 65535:  # tope uint16: se desplaza la escala de la tabla entera
                s += 1
            self.tabs.append((cnt >> s).astype(np.uint16))
            self.shifts.append(s)
            del cnt

    def n_params(self):  # celdas totales asignadas: buckets * V * M (exact: no aplica, se da nonzero)
        if self.hash == "exact":
            return int(sum(len(e[3]) for e in self.ex))
        return int(self.M * (1 << self.b) * self.V)

    def n_nonzero(self):  # celdas de contador distintas de cero
        if self.hash == "exact":
            return int(sum(len(e[3]) for e in self.ex))
        return int(sum(int((tb > 0).sum()) for tb in self.tabs))

    def _exact_counts(self, e, key, V):
        cu, ctot, typ, pk, pc = e
        j = np.minimum(np.searchsorted(cu, key), len(cu) - 1)
        found = cu[j] == key
        T = np.where(found, ctot[j], 0.0)
        U = np.where(found, typ[j], 0.0)
        kk = key[:, None] * V + np.arange(V, dtype=np.int64)[None, :]
        jj = np.minimum(np.searchsorted(pk, kk.ravel()), len(pk) - 1)
        hit = pk[jj] == kk.ravel()
        c = np.where(hit, pc[jj], 0.0).reshape(len(key), V)
        return c, T, U

    def predict(self, ids, pos):
        V = self.V
        pos = np.asarray(pos, dtype=np.int64)
        m = len(pos)
        xp = np.concatenate([np.full(self.L, V, np.uint8), ids])
        num = np.zeros((m, V)); den = np.zeros(m); logacc = np.zeros((m, V))
        q = self.q[None, :]
        if self.hash == "exact":
            parts = [(self._keys(xp, pos, offs), e, None) for offs, e in zip(self.masks, self.ex)]
        else:
            parts = [(self._buckets(self._keys(xp, pos, offs)), tb, s)
                     for offs, tb, s in zip(self.masks, self.tabs, self.shifts)]
        for key_or_bk, tb, s in parts:
            if self.hash == "exact":
                c, T, U = self._exact_counts(tb, key_or_bk, V)
            else:
                c = tb[key_or_bk].astype(np.float64) * float(1 << s)
                T = c.sum(1)
                U = (c > 0).sum(1).astype(np.float64)
            denom = T + U
            pos_d = denom > 0
            p = (c + U[:, None] * q) / np.maximum(denom, 1e-300)[:, None]
            p = np.where(pos_d[:, None], p, q)
            if self.mix == "lin_T":
                w = T
            elif self.mix == "geo_logT":  # evidencia logaritmica: log2(1+T)
                w = np.log2(1.0 + T)
            elif self.mix == "geo_seen":  # cada tabla que vio su contexto cuenta igual
                w = pos_d.astype(np.float64)
            else:
                w = np.where(pos_d, T / np.maximum(denom, 1e-300), 0.0)
            if self.mix.startswith("geo"):
                logacc += w[:, None] * np.log(np.maximum(p, 1e-300))
            else:
                num += w[:, None] * p
            den += w
        if self.mix.startswith("geo"):
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = logacc / np.maximum(den, 1e-300)[:, None]
            lg = lg - lg.max(1, keepdims=True)
            out = np.exp(lg)
            out = out / out.sum(1, keepdims=True)
        else:
            out = num / np.maximum(den, 1e-300)[:, None]
        out = np.where((den > 0)[:, None], out, q)
        out = np.maximum(out, 1e-12)
        out = out / out.sum(1, keepdims=True)
        return out.astype(np.float32)

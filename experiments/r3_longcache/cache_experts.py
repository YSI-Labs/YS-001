"""Expertos de MEMORIA A CORTO PLAZO / EN CONTEXTO para Ixchel (ronda 2, codename cache).

Sin parametros aprendidos por gradiente ni learning rate. Todo es conteo exacto + algebra cerrada.

1) CacheWB  -- conteos EXACTOS de "que siguio al contexto de orden k" dentro de una ventana del propio
   historial: ids[max(k, p-W):p]. Orden k = 1..4. Suavizado Witten-Bell jerarquico: p_k = (c + T*p_{k-1})/(n + T),
   con T = numero de simbolos distintos vistos tras ese contexto en la ventana, y base p_0 = unigrama global.
   Una instancia sirve para varias ventanas (expertos separados). Ventana infinita = todo el historial.

2) MatchIndex + match_dists -- modelo de coincidencia. Para cada posicion busca el contexto previo mas largo
   (6 o 12 caracteres, verificado hacia atras hasta MATCH_CAP) en el historial ('hist', causal: solo t < p)
   o en el train ('train', no causal: fuente fija). Predice el caracter que siguio. Confianza por conteo:
   h[bucket de longitud] = tasa de acierto medida en dev (prior de conteo prior_m). Distribucion:
   P = (1-h) * base + h * onehot(pred); sin coincidencia -> base.

CAUSALIDAD: todo valor en la posicion p depende solo de ids[:p]. Los indices se construyen con ids[:maxpos]
y las consultas filtran t < p. Verificado con leak_check del harness y con test_cache_experts.py.

Complejidad: construccion O(N log N) por llamada (argsort/sort de N claves); consulta cache O(V log N) por
orden y ventana; consulta match O(log N + cap).
"""
import numpy as np
import numba as nb

ORDERS = (1, 2, 3, 4)                          # ordenes del cache WB (contexto de k chars)
WINDOWS_DEFAULT = (256, 1024, 4096, 1 << 40)   # 1<<40 = ventana infinita (todo el historial)
MATCH_CAP = 32                                 # verificacion maxima hacia atras del match
MATCH_MIN = 6                                  # longitud minima aceptada (= clave exacta mas corta)
N_MBUCKET = 7                                  # buckets de longitud: 0 sin match, 1:[6,8) 2:[8,12) 3:[12,16) 4:[16,24) 5:[24,32) 6:>=32
TBITS = 27                                     # bits para la posicion t en la clave compuesta (N < 2**27)


# ----------------------------------------------------------------- utilidades
@nb.njit(cache=True)
def _lb(arr, lo, hi, v):
    """Primer indice idx en [lo,hi) con arr[idx] >= v (hi si no existe)."""
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


def unigram(x, V):
    c = np.bincount(np.asarray(x, dtype=np.int64), minlength=V).astype(np.float64)
    return c / max(c.sum(), 1.0)


def _keys(src, t, k):
    """Clave de los k caracteres previos a cada posicion t (t >= k): exacta si k<=6 (7 bits por char),
    hash FNV-64 truncado a 62 bits si k>6 (las colisiones se descartan con la verificacion hacia atras)."""
    src = np.asarray(src)
    t = np.asarray(t, dtype=np.int64)
    if k <= 6:
        code = np.zeros(len(t), dtype=np.int64)
        for j in range(k):
            code = code * 128 + src[t - k + j].astype(np.int64)
        return code
    h = np.full(len(t), np.uint64(1469598103934665603), dtype=np.uint64)
    for j in range(k):
        h = (h ^ (src[t - k + j].astype(np.uint64) + np.uint64(1))) * np.uint64(1099511628211)
    return (h >> np.uint64(2)).astype(np.int64)


def match_bucket(L):
    L = np.asarray(L)
    b = np.zeros(L.shape, dtype=np.int64)
    for edge, val in ((6, 1), (8, 2), (12, 3), (16, 4), (24, 5), (32, 6)):
        b[L >= edge] = val
    return b


# ----------------------------------------------------------------- 1) cache WB por ventanas
def build_cache_index(ids, maxp):
    """Para cada orden k: pares (contexto_k<<7 | siguiente) ordenados, y dentro de cada par las posiciones t
    ascendentes. Solo usa t < maxp (t = posicion del siguiente caracter)."""
    x = np.asarray(ids[:maxp]).astype(np.int64)
    N = len(x)
    PK, BS, BE, T, pko = [], [], [], [], [0]
    toff = 0
    for k in ORDERS:
        t = np.arange(k, N, dtype=np.int64)
        if len(t) == 0:
            pko.append(pko[-1])
            continue
        pk = (_keys(x, t, k) << 7) | x[t]
        comp = np.sort((pk << TBITS) | t)
        pkv = comp >> TBITS
        tt = comp & ((1 << TBITS) - 1)
        chg = np.flatnonzero(pkv[1:] != pkv[:-1]) + 1
        st = np.concatenate(([0], chg)).astype(np.int64)
        en = np.concatenate((chg, [len(pkv)])).astype(np.int64)
        PK.append(pkv[st]); BS.append(st + toff); BE.append(en + toff); T.append(tt)
        toff += len(tt)
        pko.append(pko[-1] + len(st))
    if not PK:
        z = np.zeros(0, np.int64)
        return dict(x=x, PK=z, BS=z, BE=z, T=z, pko=np.zeros(len(pko), np.int64))
    return dict(x=x, PK=np.concatenate(PK), BS=np.concatenate(BS), BE=np.concatenate(BE),
                T=np.concatenate(T), pko=np.array(pko, np.int64))


@nb.njit(cache=True)
def _wb_query(x, pos, PK, BS, BE, T, pko, windows, base, out_P, out_lvl):
    """out_P[i,w,:] = WB-4 de la ventana w en la posicion pos[i]; out_lvl[i,w] = mayor orden con conteo > 0."""
    V = base.shape[0]
    m = pos.shape[0]
    nW = windows.shape[0]
    cnt = np.zeros((nW, V))
    tot = np.zeros(nW)
    nd = np.zeros(nW)
    for i in range(m):
        p = pos[i]
        for w in range(nW):
            for v in range(V):
                out_P[i, w, v] = base[v]
            out_lvl[i, w] = 0
        for k in range(1, 5):
            if p < k:
                break
            code = 0
            for j in range(k):
                code = code * 128 + x[p - k + j]
            lo = _lb(PK, pko[k - 1], pko[k], code << 7)
            hi = _lb(PK, lo, pko[k], (code + 1) << 7)
            if hi == lo:
                break                                  # contexto de orden k nunca visto => ordenes mayores tampoco
            for w in range(nW):
                tot[w] = 0.0
                nd[w] = 0.0
                for v in range(V):
                    cnt[w, v] = 0.0
            for j in range(lo, hi):
                sym = PK[j] & 127
                s = BS[j]
                e = BE[j]
                hp = _lb(T, s, e, p)                   # entradas de este par con t < p: [s, hp)
                for w in range(nW):
                    a = p - windows[w]
                    if a < k:
                        a = k
                    c = hp - _lb(T, s, hp, a)          # entradas con t en [a, p)
                    if c > 0:
                        cnt[w, sym] = c
                        tot[w] += c
                        nd[w] += 1.0
            for w in range(nW):
                if tot[w] > 0.0:
                    td = nd[w]
                    den = tot[w] + td
                    for v in range(V):
                        out_P[i, w, v] = (cnt[w, v] + td * out_P[i, w, v]) / den
                    out_lvl[i, w] = k


class CacheWB:
    """Experto de cache: una distribucion por ventana. base = unigrama global (fijado por el llamador)."""

    def __init__(self, windows=WINDOWS_DEFAULT):
        self.windows = np.asarray(windows, dtype=np.int64)

    @property
    def nw(self):
        return len(self.windows)

    def index(self, ids, maxp):
        return build_cache_index(ids, int(maxp))

    def query(self, idx, pos, base):
        pos = np.asarray(pos, dtype=np.int64)
        V = len(base)
        out_P = np.empty((len(pos), self.nw, V), dtype=np.float64)
        out_lvl = np.zeros((len(pos), self.nw), dtype=np.int64)
        if idx["PK"].size == 0:
            out_P[:] = np.asarray(base, np.float64)[None, None, :]
            return out_P, out_lvl
        _wb_query(idx["x"], pos, idx["PK"], idx["BS"], idx["BE"], idx["T"], idx["pko"],
                  self.windows, np.asarray(base, np.float64), out_P, out_lvl)
        return out_P, out_lvl


# ----------------------------------------------------------------- 2) match model
def build_hash_index(src, N, k):
    """Pares (clave de contexto k, t) ordenados por clave y, a igual clave, t ascendente. t en [k, N)."""
    t = np.arange(k, N, dtype=np.int64)
    if len(t) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    key = _keys(src, t, k)
    order = np.argsort(key, kind="stable")
    return key[order], t[order]


@nb.njit(cache=True)
def _match_query(ids, pos, qk6, qk12, K6, T6, K12, T12, src, causal, cap, minl, out_sym, out_L):
    m = pos.shape[0]
    for i in range(m):
        p = pos[i]
        bestL = 0
        bests = -1
        for which in range(2):
            if which == 0:
                K = K12; T = T12; key = qk12[i]; kk = 12
            else:
                K = K6; T = T6; key = qk6[i]; kk = 6
            if p < kk or K.shape[0] == 0:
                continue
            lo = _lb(K, 0, K.shape[0], key)
            hi = _lb(K, lo, K.shape[0], key + 1)
            if hi <= lo:
                continue
            if causal:
                idx = _lb(T, lo, hi, p) - 1           # la mayor posicion t < p con esa clave
                if idx < lo:
                    continue
            else:
                idx = hi - 1                          # fuente fija (train): ultima aparicion
            t = T[idx]
            L = 0
            while L < cap and t - 1 - L >= 0 and p - 1 - L >= 0 and src[t - 1 - L] == ids[p - 1 - L]:
                L += 1
            if L >= minl and L > bestL:
                bestL = L
                bests = src[t]
        out_sym[i] = bests
        out_L[i] = bestL


class MatchIndex:
    """Indice de coincidencias sobre src[:N]. causal=True: se usa para el historial (solo t < p);
    causal=False: fuente fija (train)."""

    def __init__(self, src, N, causal):
        self.src = np.asarray(src[:int(N)]).astype(np.int64)
        self.causal = bool(causal)
        N = len(self.src)
        self.K6, self.T6 = build_hash_index(self.src, N, 6)
        self.K12, self.T12 = build_hash_index(self.src, N, 12)

    def raw(self, ids, pos):
        """Devuelve (sym[m], L[m]): caracter predicho (-1 si no hay coincidencia >= MATCH_MIN) y longitud verificada."""
        pos = np.asarray(pos, dtype=np.int64)
        x = np.asarray(ids).astype(np.int64)
        qk6 = _keys(x, np.maximum(pos, 6), 6)
        qk12 = _keys(x, np.maximum(pos, 12), 12)
        sym = np.full(len(pos), -1, dtype=np.int64)
        L = np.zeros(len(pos), dtype=np.int64)
        _match_query(x, pos, qk6, qk12, self.K6, self.T6, self.K12, self.T12, self.src, self.causal,
                     MATCH_CAP, MATCH_MIN, sym, L)
        return sym, L


def match_hits(sym, L, x):
    """Conteos por bucket sobre posiciones con coincidencia: (aciertos[7], total[7])."""
    b = match_bucket(L)
    v = sym >= 0
    tot = np.bincount(b[v], minlength=N_MBUCKET).astype(np.float64)
    hit = np.bincount(b[v & (sym == x)], minlength=N_MBUCKET).astype(np.float64)
    return hit, tot


def match_accuracy(hit, tot, prior_m=1.0):
    """Tasa de acierto por bucket con prior de conteo: (hit + prior_m * r) / (tot + prior_m), r = tasa global."""
    r = hit.sum() / max(tot.sum(), 1.0)
    return (hit + prior_m * r) / (tot + prior_m)


def match_dists(sym, L, base, h):
    """P[m,V] = (1-h_b) base + h_b onehot(pred) con h_b = h[bucket(L)]; sin coincidencia -> base."""
    m, V = len(sym), len(base)
    P = np.tile(np.asarray(base, np.float64), (m, 1))
    v = np.flatnonzero(sym >= 0)
    hb = np.asarray(h, np.float64)[match_bucket(L)][v]
    P[v] = (1.0 - hb)[:, None] * np.asarray(base, np.float64)[None, :]
    P[v, sym[v]] += hb
    return P

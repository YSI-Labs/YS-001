"""r3_longcache: expertos de MEMORIA EN CONTEXTO generalizados (sin gradientes, sin learning rate).

Un KeyedWin es una cadena jerarquica Witten-Bell sobre niveles de contexto (coarse -> fine). Cada nivel tiene una
clave por posicion K_l[t] calculada SOLO desde ids[:t] (causal) y el simbolo x[t] = ids[t]. Niveles disponibles:
  ('c', k)  : los k caracteres previos (codigo exacto, b bits por caracter, k*b <= 62)
  ('w',)    : hash del prefijo de la palabra actual (wp; 0 al inicio de palabra)
  ('w2',)   : hash de (palabra previa completa pw1, wp)
Modos de ponderacion de la historia:
  mode 0 'win' : conteos dentro de una ventana dura [p-W, p)  (W=inf -> todo el historial)
  mode 1 'dec' : conteos con decaimiento exponencial exp(-(p-t)/tau), EXACTO (recursion por grupo)
Salida: para cada ventana/tau, una distribucion [m,V] float64 (normalizada).

INDICE UNICO (exacto, incremental en el sentido de construir una vez): el indice se construye sobre el array
COMPLETO (p.ej. val entero) y las consultas filtran t < p con busqueda binaria. Como K_l[t] y x[t] solo
dependen de ids[:t+1], las entradas con t >= p no influyen en la prediccion en p (verificado con leak_check).
Cache por contenido (len + crc32) para no reconstruir en predicciones repetidas sobre el mismo array.
"""
import zlib
import numpy as np
import numba as nb


# ------------------------------------------------------------------ claves
def _char_bits(V):
    return int(np.ceil(np.log2(V + 1)))


def char_keys(ids, k, b):
    """K[t] = codigo de ids[t-k:t] (pad = 0, caracter c -> c+1). Exacto si k*b <= 62."""
    assert k * b <= 62, "orden demasiado alto para clave exacta"
    N = len(ids)
    xp = np.zeros(N + 8, dtype=np.int64)
    xp[8:] = ids.astype(np.int64) + 1
    K = np.zeros(N, dtype=np.int64)
    for j in range(1, k + 1):
        K |= xp[8 - j: 8 - j + N] << np.int64(b * (j - 1))
    return K


@nb.njit(cache=True)
def _word_state(ids, isletter):
    n = ids.shape[0]
    wp = np.empty(n, np.uint64)
    pw = np.empty(n, np.uint64)
    h = np.uint64(0)
    prev = np.uint64(0)
    wl = 0
    P = np.uint64(1099511628211)
    for t in range(n):
        wp[t] = h
        pw[t] = prev
        c = ids[t]
        if isletter[c] == 1:
            h = (h + np.uint64(c) + np.uint64(1)) * P
            wl += 1
        else:
            if wl > 0:
                prev = h
            h = np.uint64(0)
            wl = 0
    return wp, pw


def word_keys(ids, isletter):
    wp, pw = _word_state(ids, isletter)
    k2 = (pw * np.uint64(0x9E3779B97F4A7C15)) ^ (wp + np.uint64(0x632BE59BD9B4E019))
    k2 ^= k2 >> np.uint64(31)
    k2 *= np.uint64(0xBF58476D1CE4E5B9)
    k2 ^= k2 >> np.uint64(29)
    return {"w": (wp >> np.uint64(1)).astype(np.int64), "w2": (k2 >> np.uint64(1)).astype(np.int64)}


# ------------------------------------------------------------------ indice por nivel
def _build_level(K, x):
    N = len(K)
    order = np.lexsort((x, K)).astype(np.int32)          # estable: t ascendente dentro de (K, x)
    Ks = K[order]
    xs = x[order]
    chg = np.flatnonzero((Ks[1:] != Ks[:-1]) | (xs[1:] != xs[:-1])) + 1
    sub_s = np.concatenate(([0], chg)).astype(np.int32)
    sub_e = np.concatenate((chg, [N])).astype(np.int32)
    sub_x = xs[sub_s].astype(np.int64)
    sub_K = Ks[sub_s]
    gchg = np.flatnonzero(sub_K[1:] != sub_K[:-1]) + 1
    ugs = np.concatenate(([0], gchg, [len(sub_s)])).astype(np.int64)
    UK = sub_K[ugs[:-1]].astype(np.int64)
    return dict(UK=UK, ugs=ugs, sub_x=sub_x, sub_s=sub_s, sub_e=sub_e, ts=order, K=K)


@nb.njit(cache=True)
def _decay_R(ts, sub_s, sub_e, tau, R):
    # R[j] = sum_{i<=j, mismo grupo} exp(-(ts[j]-ts[i])/tau)  (recursion exacta)
    for q in range(sub_s.shape[0]):
        s = sub_s[q]
        e = sub_e[q]
        R[s] = 1.0
        for j in range(s + 1, e):
            R[j] = 1.0 + R[j - 1] * np.exp(-(ts[j] - ts[j - 1]) / tau)


@nb.njit(cache=True)
def _lbi(arr, lo, hi, v):
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@nb.njit(cache=True)
def _lc_query(UK, ugs, sub_x, sub_s, sub_e, ts, R, mode, params, qK, pos, P, lvl, lidx):
    """Actualiza P[w,i,:] (WB de este nivel sobre la distribucion gruesa previa) para cada ventana/tau w."""
    nw = params.shape[0]
    m = pos.shape[0]
    V = P.shape[2]
    G = UK.shape[0]
    tmp = np.zeros(V)
    for i in range(m):
        p = pos[i]
        g = _lbi(UK, 0, G, qK[i])
        if g >= G or UK[g] != qK[i]:
            continue
        for w in range(nw):
            for v in range(V):
                tmp[v] = 0.0
            n = 0.0
            T = 0.0
            for q in range(ugs[g], ugs[g + 1]):
                s = sub_s[q]
                e = sub_e[q]
                hi = _lbi(ts, s, e, p)               # primer indice con ts >= p
                if hi == s:
                    continue
                if mode == 0:
                    a = p - params[w]
                    lo = _lbi(ts, s, hi, a)
                    mass = float(hi - lo)
                    if mass <= 0.0:
                        continue
                else:
                    j = hi - 1
                    mass = R[w, j] * np.exp(-(p - ts[j]) / params[w])
                tmp[sub_x[q]] += mass
                n += mass
                T += 1.0
            if n > 0.0:
                den = n + T
                for v in range(V):
                    P[w, i, v] = (tmp[v] + T * P[w, i, v]) / den
                lvl[w, i] = lidx + 1


class KeyedWin:
    """Cadena WB jerarquica sobre niveles de contexto con ventanas duras (mode='win') o decaimiento (mode='dec').

    levels: lista coarse->fine de especificaciones: ('c',k) | ('w',) | ('w2',).
    windows: lista de ventanas en chars (1<<40 = todo el historial)  [mode='win']
    tau: vida del decaimiento en chars [mode='dec'] (si tau is None se elige por max-verosimilitud en dev)."""

    def __init__(self, name, levels, windows=None, mode="win", tau=None):
        self.name = name
        self.levels = list(levels)
        self.mode = 0 if mode == "win" else 1
        if self.mode == 0:
            self.params = np.asarray(windows if windows is not None else [1 << 40], dtype=np.float64)
        else:
            self.params = np.asarray([tau if tau is not None else 1000.0], dtype=np.float64)
        self.tau_fixed = tau is not None or self.mode == 0
        self.isletter = None
        self.V = None
        self.b = None
        self._cache_key = None
        self._idx = None
        self._R = None

    @property
    def nw(self):
        return len(self.params)

    def columns(self):
        if self.mode == 0:
            return [f"{self.name}_W{int(w) if w < (1 << 40) else 'inf'}" for w in self.params]
        return [f"{self.name}_tau{self.params[0]:.0f}"]

    def set_vocab(self, V, isletter):
        self.V = V
        self.isletter = isletter
        self.b = _char_bits(V)

    def _build(self, ids):
        keys = {}
        need_word = any(l[0] in ("w", "w2") for l in self.levels)
        if need_word:
            keys = word_keys(ids, self.isletter)
        x = ids.astype(np.int64)
        idx = []
        for l in self.levels:
            K = char_keys(ids, l[1], self.b) if l[0] == "c" else keys[l[0]]
            idx.append(_build_level(K, x))
        return idx

    def prepare(self, ids):
        """Indice sobre el array completo `ids` (cacheado por contenido: longitud + crc32)."""
        key = (len(ids), zlib.crc32(np.ascontiguousarray(ids).tobytes()))
        if self._cache_key != key:
            self._idx = self._build(ids)
            self._cache_key = key
            self._R = None
        return self

    def set_tau(self, tau):
        self.params = np.asarray([tau], dtype=np.float64)
        self._R = None

    def _decay_arrays(self):
        if self._R is None:
            tau = float(self.params[0])
            self._R = []
            for L in self._idx:
                R = np.empty(len(L["ts"]), dtype=np.float64)
                _decay_R(L["ts"], L["sub_s"], L["sub_e"], tau, R)
                self._R.append(R)
        return self._R

    def query(self, pos, base):
        """Devuelve (lista de nw arrays [m,V] float64, lvl [nw,m] = nivel mas fino con datos)."""
        pos = np.asarray(pos, dtype=np.int64)
        base = np.asarray(base, dtype=np.float64)
        m, V = len(pos), len(base)
        P = np.broadcast_to(base, (self.nw, m, V)).copy()
        lvl = np.zeros((self.nw, m), dtype=np.int64)
        Rs = self._decay_arrays() if self.mode == 1 else None
        for li, L in enumerate(self._idx):
            R = Rs[li][None, :] if self.mode == 1 else np.zeros((1, 1))
            qK = L["K"][pos]
            _lc_query(L["UK"], L["ugs"], L["sub_x"], L["sub_s"], L["sub_e"], L["ts"], R, self.mode,
                      self.params, qK, pos, P, lvl, li)
        return [P[w] for w in range(self.nw)], lvl

    def nll_dev(self, pos, xs, base):
        Ps, _ = self.query(pos, base)
        return float(-np.log2(np.maximum(Ps[0][np.arange(len(pos)), xs], 1e-12)).mean())

    def select_tau(self, pos, xs, base, lo=8.0, hi=2e5, iters=16, grid=None, verbose=True):
        """Maximo de verosimilitud en dev (1-D, busqueda de oro sobre log tau). Sin learning rate."""
        if grid is not None and verbose:
            curve = {float(t): round(self._nll_at(t, pos, xs, base), 4) for t in grid}
            print(f"  [{self.name}] curva dev (tau -> bpc experto solo): {curve}", flush=True)
        a, b = np.log(lo), np.log(hi)
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = self._nll_at(np.exp(c), pos, xs, base), self._nll_at(np.exp(d), pos, xs, base)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = self._nll_at(np.exp(c), pos, xs, base)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = self._nll_at(np.exp(d), pos, xs, base)
        tau = float(np.exp((a + b) / 2))
        self.set_tau(tau)
        if verbose:
            print(f"  [{self.name}] tau elegido por dev-ML = {tau:.1f} chars (dev bpc experto solo {min(fc, fd):.4f})", flush=True)
        return tau

    def _nll_at(self, tau, pos, xs, base):
        self.set_tau(tau)
        return self.nll_dev(pos, xs, base)

"""r4_loops: recetas de GENERACION sobre Ixchel v0.4 (base fijada, sin re-ajustar nada).

Kernel propio del experto de cache (KeyedWin modo win) con DOS rangos temporales permitidos [lo1,hi1) U [lo2,hi2)
por posicion, para poder excluir caracteres generados del historial. Con rangos [max(0,p-W),p) reproduce
KeyedWin.query exactamente (verificado en check_equiv()).

Recetas (P = longitud del prompt, p = posicion a predecir):
  base : historial = todo lo previo (comportamiento de v0.4).
  a    : historial = solo el prompt (ventanas W aplicadas al prompt). Chars generados NO entran al cache.
  b    : historial = prompt + ultimos 256 chars generados (256 = ventana ya existente en la base).
  c1   : base, pero el peso PoE de los expertos de cache se multiplica por P/p (fraccion del contexto que es prompt).
  c2   : base, pero el peso PoE de los expertos de cache se multiplica por (1 - tasa de 4-gramas repetidos del texto generado).
  d    : base, pero un experto de cache se retira de la mezcla en la posicion p si su argmax coincide con el caracter
         inmediatamente anterior (repeticion inmediata del ultimo caracter).
Los parametros de muestreo (T, top-p, penalizacion) se pasan aparte: son COMPARACION etiquetada, no receta.
"""
import os, sys, re
import numpy as np, numba as nb

ROOT = __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ixchel"))
from longcache import _lbi  # busqueda binaria (numba) reutilizada del modulo original


@nb.njit(cache=False)
def _count_range(ts, s, e, lo, hi):
    if hi <= lo:
        return 0
    return _lbi(ts, s, e, hi) - _lbi(ts, s, e, lo)


@nb.njit(cache=False)
def _q2(UK, ugs, sub_x, sub_s, sub_e, ts, qK, lo1, hi1, lo2, hi2, P, lvl, lidx):
    """Actualiza P[i,:] (WB de este nivel sobre el P previo) con el conteo en [lo1,hi1) U [lo2,hi2) para cada consulta i."""
    G = UK.shape[0]; m = qK.shape[0]; V = P.shape[1]
    tmp = np.zeros(V)
    for i in range(m):
        g = _lbi(UK, 0, G, qK[i])
        if g >= G or UK[g] != qK[i]:
            continue
        for v in range(V):
            tmp[v] = 0.0
        n = 0.0; T = 0.0
        for q in range(ugs[g], ugs[g + 1]):
            s = sub_s[q]; e = sub_e[q]
            mass = float(_count_range(ts, s, e, lo1[i], hi1[i]) + _count_range(ts, s, e, lo2[i], hi2[i]))
            if mass <= 0.0:
                continue
            tmp[sub_x[q]] += mass; n += mass; T += 1.0
        if n > 0.0:
            den = n + T
            for v in range(V):
                P[i, v] = (tmp[v] + T * P[i, v]) / den
            lvl[i] = lidx + 1


def windows_for(recipe, xp, pos, Pp):
    """Lista (uno por ventana de xp) de (lo1,hi1,lo2,hi2) arrays [m] segun la receta."""
    pos = np.asarray(pos, np.int64); Pp = np.asarray(Pp, np.int64)
    res = []
    for W in xp.params:
        W = int(min(W, 1 << 40))
        if recipe in ("base", "c1", "c2", "d"):
            lo1 = np.maximum(0, pos - W); hi1 = pos.copy()
            lo2 = np.zeros_like(pos); hi2 = np.zeros_like(pos)
        else:
            q = np.minimum(pos, Pp)                          # historial limitado al prompt
            lo1 = np.maximum(0, q - W); hi1 = q
            if recipe == "a":
                lo2 = np.zeros_like(pos); hi2 = np.zeros_like(pos)
            elif recipe == "b":
                lo2 = np.maximum(Pp, pos - 256); hi2 = np.maximum(lo2, pos)
            else:
                raise ValueError(recipe)
        res.append((lo1, hi1, lo2, hi2))
    return res


def cache_expert_dists(xp, pos, Pp, recipe, base):
    """nw columnas del cache [m,V] para la receta. xp.prepare(ids) debe haberse llamado."""
    wins = windows_for(recipe, xp, pos, Pp)
    m = len(pos); V = len(base)
    outs = []
    for (lo1, hi1, lo2, hi2) in wins:
        P = np.broadcast_to(np.asarray(base, np.float64), (m, V)).copy()
        lvl = np.zeros(m, np.int64)
        for li, L in enumerate(xp._idx):
            qK = L["K"][pos]
            _q2(L["UK"], L["ugs"], L["sub_x"], L["sub_s"], L["sub_e"], L["ts"], qK, lo1, hi1, lo2, hi2, P, lvl, li)
        outs.append(P)
    return outs


def predict_recipe(model, ids, pos, Pp, recipe, gen_rep=None):
    """Distribucion [m,V] float32 de Ixchel v0.4 bajo la receta. ids: uint8 (len >= max(pos)+1).
    Pp: longitud del prompt por posicion (Pp == pos => no hay texto generado => identidad con base).
    gen_rep: [m] tasa de 4-gramas repetidos del texto generado (solo receta c2)."""
    pos = np.asarray(pos, np.int64); Pp = np.asarray(Pp, np.int64); m = len(pos)
    V = model.V
    A = model._atoms_for(ids)
    Pch, _, _ = model._chain_dists(model.tabs, A, pos, V)
    xp = model.xperts[0]; xp.prepare(ids)
    Pc = cache_expert_dists(xp, pos, Pp, recipe, model.base)
    Ps = list(Pch) + Pc
    nch, nc = len(Pch), len(Pc)
    W = np.tile(np.asarray(model.poe_w, np.float64), (m, 1))   # [m,K]
    if recipe == "c1":
        f = np.where(pos > 0, Pp / np.maximum(pos, 1), 1.0)
        W[:, nch:] *= f[:, None]
    elif recipe == "c2":
        gr = np.zeros(m) if gen_rep is None else np.asarray(gen_rep, np.float64)
        W[:, nch:] *= (1.0 - gr)[:, None]
    elif recipe == "d":
        c1 = ids[np.maximum(pos - 1, 0)].astype(np.int64)
        for j in range(nc):
            am = np.argmax(Pc[j], axis=1)
            W[:, nch + j] = np.where((am == c1) & (pos > 0), 0.0, W[:, nch + j])
    L = np.zeros((m, V))
    for k in range(len(Ps)):
        L += W[:, k:k + 1] * np.log(np.maximum(Ps[k], 1e-12))
    L -= L.max(1, keepdims=True)
    p = np.exp(L)
    return (p / p.sum(1, keepdims=True)).astype(np.float32)


# ------------------------------------------------------------ muestreo y generacion
def sample_char(p, rng, T=1.0, top_p=None, penalty=None, recent=None):
    q = p.astype(np.float64)
    if penalty is not None and recent is not None:          # COMPARACION: penalizacion de repeticion (hiperparametro)
        for c in set(recent):
            if q[c] > 0:
                q[c] = q[c] / penalty
    if T != 1.0:
        q = q ** (1.0 / T)
    q /= q.sum()
    if top_p is not None:                                    # COMPARACION: nucleo (hiperparametro)
        order = np.argsort(-q); cs = np.cumsum(q[order])
        keep = order[: int(np.searchsorted(cs, top_p) + 1)]
        mask = np.zeros_like(q); mask[keep] = 1.0; q = q * mask; q /= q.sum()
    return int(rng.choice(len(q), p=q))


def words(s):
    return re.findall(r"[a-z]+|\d+|[^\sa-z\d]", s.lower())


def rep4_rate_text(s):
    """Fraccion de 4-gramas de palabras que repiten uno anterior (0 si hay menos de 5 palabras)."""
    w = words(s)
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    if not g:
        return 0.0
    return 1.0 - len(set(g)) / len(g)


def generate(model, prompt_ids, n_chars, recipe, rng, T=1.0, top_p=None, penalty=None):
    """Genera n_chars caracteres (ids uint8) a partir de prompt_ids."""
    P = len(prompt_ids)
    buf = np.zeros(P + n_chars + 1, dtype=np.uint8); buf[:P] = prompt_ids
    L = P
    for _ in range(n_chars):
        gen_rep = None
        if recipe == "c2":
            gen_rep = np.array([rep4_rate_text("".join(model.itos[i] for i in buf[P:L]))])
        p = predict_recipe(model, buf[:L + 1], np.array([L]), np.array([P]), recipe, gen_rep=gen_rep)[0]
        recent = buf[max(P, L - 100):L].tolist() if penalty is not None else None
        buf[L] = sample_char(p, rng, T=T, top_p=top_p, penalty=penalty, recent=recent); L += 1
    return buf[P:L].copy()


def check_equiv(model, val, n=300, seed=0):
    """max |predict_recipe('base') - model.predict| en n posiciones de val (debe ser ~0)."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(64, len(val) - 1, size=n).astype(np.int64)
    ref = model.predict(val, pos).astype(np.float64)
    got = predict_recipe(model, val, pos, pos.copy(), "base").astype(np.float64)
    return float(np.abs(ref - got).max())

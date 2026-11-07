"""Test de correccion de cache_experts.py contra fuerza bruta (Python puro) + causalidad.
Uso: python -I test_cache_experts.py"""
import os, sys
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from cache_experts import (CacheWB, MatchIndex, match_dists, match_hits, match_accuracy, unigram)

WINDOWS = (16, 64, 1 << 40)


def brute_cache(ids, p, V, base, W):
    out = np.array(base, float)
    for k in range(1, 5):
        if p < k:
            break
        ctx = tuple(ids[p - k:p])
        cnt = np.zeros(V);
        for t in range(max(k, p - W), p):
            if tuple(ids[t - k:t]) == ctx:
                cnt[ids[t]] += 1
        tot = cnt.sum()
        if tot > 0:
            Td = (cnt > 0).sum()
            out = (cnt + Td * out) / (tot + Td)
    return out


def brute_match(ids, p, src, causal):
    best_L, best_s = 0, -1
    for kk in (12, 6):
        if p < kk:
            continue
        ctx = tuple(ids[p - kk:p])
        if causal:
            cands = [t for t in range(kk, p) if tuple(src[t - kk:t]) == ctx]
        else:
            cands = [t for t in range(kk, len(src)) if tuple(src[t - kk:t]) == ctx]
        if not cands:
            continue
        t = cands[-1]
        L = 0
        while L < 32 and t - 1 - L >= 0 and p - 1 - L >= 0 and src[t - 1 - L] == ids[p - 1 - L]:
            L += 1
        if L >= 6 and L > best_L:
            best_L, best_s = L, int(src[t])
    return best_s, best_L


def main():
    rng = np.random.default_rng(0)
    V = 4
    # secuencia con repeticiones: bloques copiados + ruido para forzar coincidencias largas
    base_block = rng.integers(0, V, 60)
    ids = np.concatenate([rng.integers(0, V, 40), base_block, rng.integers(0, V, 30), base_block[:40],
                          rng.integers(0, V, 200)]).astype(np.uint8)
    N = len(ids)
    uni = unigram(ids, V)
    cw = CacheWB(WINDOWS)
    idx = cw.index(ids, N)
    pos = np.arange(1, N, 3, dtype=np.int64)
    P, lvl = cw.query(idx, pos, uni)
    err = 0.0
    for i, p in enumerate(pos):
        for w, W in enumerate(WINDOWS):
            ref = brute_cache(ids, int(p), V, uni, W)
            err = max(err, float(np.abs(P[i, w] - ref).max()))
    print(f"[cache WB] max |numba - bruta| = {err:.2e}  (puntos {len(pos)}, ventanas {WINDOWS})")
    assert err < 1e-9, "cache WB no coincide con fuerza bruta"

    # causalidad: cambiar el futuro no altera la prediccion
    ids2 = ids.copy(); ids2[100:] = rng.integers(0, V, N - 100)
    P2, _ = cw.query(cw.index(ids2, 100 + 1), np.array([50, 99, 100]), uni)
    P1, _ = cw.query(cw.index(ids, 101), np.array([50, 99, 100]), uni)
    print(f"[cache WB] causal (futuro cambiado tras 100): {np.allclose(P1[:2], P2[:2], atol=1e-12)}")
    assert np.allclose(P1[:2], P2[:2], atol=1e-12)

    # match: historial (causal) y train (fuente fija)
    mi = MatchIndex(ids, N, causal=True)
    sym, L = mi.raw(ids, pos)
    bad = 0
    for i, p in enumerate(pos):
        s_ref, L_ref = brute_match(ids, int(p), ids, True)
        if (s_ref, L_ref) != (int(sym[i]), int(L[i])) and not (L_ref < 6 and sym[i] < 0):
            bad += 1
    print(f"[match hist] discrepancias con fuerza bruta: {bad} de {len(pos)}")
    assert bad == 0

    src = rng.integers(0, V, 300).astype(np.uint8)
    src[120:180] = base_block[:60]
    ti = MatchIndex(src, len(src), causal=False)
    qpos = np.arange(70, N, 7)
    sym_t, L_t = ti.raw(ids, qpos)
    bad_t = 0
    for i, p in enumerate(qpos):
        s_ref, L_ref = brute_match(ids, int(p), src, False)
        if (s_ref, L_ref) != (int(sym_t[i]), int(L_t[i])):
            bad_t += 1
    print(f"[match train] discrepancias con fuerza bruta: {bad_t} de {len(qpos)}")
    assert bad_t == 0

    # distribuciones del match: normalizadas y positivas
    h = match_accuracy(*match_hits(sym, L, ids[pos]))
    Pm = match_dists(sym, L, uni, h)
    print(f"[match dists] suma max error = {np.abs(Pm.sum(1)-1).max():.1e}, min p = {Pm.min():.3e}")
    assert np.abs(Pm.sum(1) - 1).max() < 1e-9 and Pm.min() > 0
    print("OK")


if __name__ == "__main__":
    main()

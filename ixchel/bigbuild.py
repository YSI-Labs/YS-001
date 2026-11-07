"""Construccion FUERA DE MEMORIA de las tablas de Ixchel por PARTICIONES DE RANGO DE CLAVE (para modelos de ~10^9 pares).

Idea: las claves de par pk = (hash_contexto & ~127) | simbolo estan bien mezcladas, asi que el rango [0, 2^64) se parte en
NPART intervalos contiguos (bits altos de pk). Cada particion se construye por separado y la concatenacion en orden da los
arrays GLOBALMENTE ORDENADOS que usa `_chain_predict`. Se escanea el corpus en trozos (chunk) por GRUPOS de particiones:
  - nivel mas fino de cada cadena: conteos CRUDOS (np.unique por trozo + suma entre trozos);
  - niveles inferiores: conteos de CONTINUACION (KN): numero de pares hijo distintos por par padre
    (se guardan pares (pk_hijo, pk_padre) unicos; la particion se decide por la clave del PADRE).
Equivalencia exacta con `Ixchel._count` verificada en test_bigbuild.py (un solo trozo -> tablas identicas).
Aproximacion declarada: los atomos de palabra/linea se reinician en cada trozo (los primeros ~16 caracteres de cada trozo se omiten)."""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core_v04 import MASK, _make_table, _concat_tables

LOW7 = np.uint64(127)


def _split_sorted(U, C, shift, g0, group):
    """U ordenado (uint64) con valores por particion g0..g0+group-1; devuelve lista [(p, U_p, C_p)]."""
    pp = (U >> np.uint64(shift)).astype(np.int64)
    out = []
    for p in range(g0, g0 + group):
        lo = np.searchsorted(pp, p, side="left"); hi = np.searchsorted(pp, p, side="right")
        out.append((p, U[lo:hi], None if C is None else C[lo:hi]))
    return out


def _reduce_counts(Ulist, Clist):
    U = np.concatenate(Ulist); C = np.concatenate(Clist)
    order = np.argsort(U, kind="stable"); U = U[order]; C = C[order]
    if len(U) == 0:
        return U, C
    brk = np.flatnonzero(U[1:] != U[:-1]) + 1
    starts = np.concatenate(([0], brk))
    return U[starts], np.add.reduceat(C, starts)


def _spill_scan(model, train, cut, spill, names, chunk, t0, verbose):
    """FASE 1: un solo escaneo del corpus; por trozo y nivel vuelca a disco (sin particionar) los pares unicos."""
    L1 = {nm: len(model.chain_idx[nm][1]) - 1 for nm in names}
    os.makedirs(spill, exist_ok=True)
    nchunks = 0
    T0 = time.time()
    for ci, c0 in enumerate(range(0, cut, chunk)):
        c1 = min(c0 + chunk, cut)
        ids = np.asarray(train[c0:c1])
        if len(ids) <= t0 + 1:
            continue
        A = model._atoms_for(ids)
        pos = np.arange(t0, len(ids), dtype=np.int64)
        xs = ids[t0:].astype(np.uint64)
        for nm in names:
            H = model._H(A, pos, nm)
            top = L1[nm] - 1
            for l in range(top, -1, -1):
                pk = (H[l] & MASK) | xs
                if l == top:
                    u, c = np.unique(pk, return_counts=True)
                    np.save(os.path.join(spill, f"{nm}_top_u_{ci}.npy"), u); np.save(os.path.join(spill, f"{nm}_top_c_{ci}.npy"), c.astype(np.uint32))
                if l >= 1:
                    ppk = (H[l - 1] & MASK) | xs
                    u2, idx = np.unique(pk, return_index=True)
                    pp = ppk[idx]; order = np.argsort(pp, kind="stable")
                    np.save(os.path.join(spill, f"{nm}_L{l-1}_pp_{ci}.npy"), pp[order]); np.save(os.path.join(spill, f"{nm}_L{l-1}_pk_{ci}.npy"), (u2[order] >> np.uint64(32)).astype(np.uint32))
            del H
        del A
        nchunks += 1
        if verbose:
            print(f"[bigbuild scan {','.join(names)}] trozo {ci} ({c1/1e6:.0f}M/{cut/1e6:.0f}M chars) t={time.time()-T0:.0f}s", flush=True)
    return nchunks


def _bounds(arr, p, nparts, shift):
    lo = np.searchsorted(arr, np.uint64(p) << np.uint64(shift), side="left")
    hi = len(arr) if p == nparts - 1 else np.searchsorted(arr, np.uint64(p + 1) << np.uint64(shift), side="left")
    return lo, hi


def _spill_reduce(model, spill, outdir, names, nparts, nchunks, verbose):
    """FASE 2: por cadena/nivel/particion lee los trozos volcados, reduce, y ensambla las tablas finales."""
    import math
    shift = 64 - int(math.log2(nparts))
    L1 = {nm: len(model.chain_idx[nm][1]) - 1 for nm in names}
    meta = {}
    for nm in names:
        tabs = []
        for l in range(L1[nm]):
            Us, Cs = [], []
            for p in range(nparts):
                if l == L1[nm] - 1:
                    ul, cl = [], []
                    for ci in range(nchunks):
                        fu = os.path.join(spill, f"{nm}_top_u_{ci}.npy")
                        if not os.path.exists(fu):
                            continue
                        u = np.load(fu, mmap_mode="r"); lo, hi = _bounds(u, p, nparts, shift)
                        if hi > lo:
                            ul.append(np.asarray(u[lo:hi])); cl.append(np.asarray(np.load(os.path.join(spill, f"{nm}_top_c_{ci}.npy"), mmap_mode="r")[lo:hi]).astype(np.int64))
                    if ul:
                        U, C = _reduce_counts(ul, cl)
                    else:
                        U, C = np.zeros(0, np.uint64), np.zeros(0, np.int64)
                else:
                    ppl, pkl = [], []
                    for ci in range(nchunks):
                        fp = os.path.join(spill, f"{nm}_L{l}_pp_{ci}.npy")
                        if not os.path.exists(fp):
                            continue
                        pp = np.load(fp, mmap_mode="r"); lo, hi = _bounds(pp, p, nparts, shift)
                        if hi > lo:
                            ppl.append(np.asarray(pp[lo:hi])); pkl.append(np.asarray(np.load(os.path.join(spill, f"{nm}_L{l}_pk_{ci}.npy"), mmap_mode="r")[lo:hi]))
                    if ppl:
                        PP = np.concatenate(ppl); PK = np.concatenate(pkl)
                        order = np.lexsort((PK, PP)); PK = PK[order]; PP = PP[order]          # pares (padre, hijo32) unicos
                        first = np.concatenate(([True], (PK[1:] != PK[:-1]) | (PP[1:] != PP[:-1]))); PP = PP[first]
                        brk = np.flatnonzero(PP[1:] != PP[:-1]) + 1; starts = np.concatenate(([0], brk))
                        U = PP[starts]; C = np.diff(np.concatenate((starts, [len(PP)]))).astype(np.int64)
                    else:
                        U, C = np.zeros(0, np.uint64), np.zeros(0, np.int64)
                Us.append(U); Cs.append(C)
            tabs.append(_make_table(np.concatenate(Us), np.concatenate(Cs)))
            del Us, Cs
        T = _concat_tables(tabs)
        d = os.path.join(outdir, nm); os.makedirs(d, exist_ok=True)
        for k, v in T.items():
            np.save(os.path.join(d, k + ".npy"), v)
        meta[nm] = dict(levels=L1[nm], pairs=int(len(T["psym"])), contexts=int(len(T["ckey"])))
        if verbose:
            print(f"[bigbuild reduce] cadena {nm}: {meta[nm]}", flush=True)
        del tabs, T
    return meta


def build_chains_big(model, train, cut, outdir, nparts=16, chunk=20_000_000, chains=None, t0=16, verbose=True, keep_spill=False, group=None):
    """Fase 1 (escaneo+volcado) + fase 2 (reduccion por particion + ensamblaje). Procesos distintos pueden construir cadenas
    distintas (chains=[...]) con el mismo outdir. Devuelve meta con tiempos."""
    names = chains or model.names
    os.makedirs(outdir, exist_ok=True)
    spill = os.path.join(outdir, "_spill_" + "_".join(names))
    T0 = time.time()
    nchunks = len(range(0, cut, chunk))
    _spill_scan(model, train, cut, spill, names, chunk, t0, verbose)
    t1 = time.time()
    meta = _spill_reduce(model, spill, outdir, names, nparts, nchunks, verbose)
    t2 = time.time()
    meta["_stats"] = dict(scan_s=t1 - T0, reduce_s=t2 - t1, total_s=t2 - T0, chunks=nchunks)
    meta["_cfg"] = dict(nparts=nparts, chunk=chunk, cut=int(cut), t0=t0, chains=list(names))
    json.dump(meta, open(os.path.join(outdir, "meta_" + "_".join(names) + ".json"), "w"), indent=1)
    if not keep_spill:
        import shutil
        shutil.rmtree(spill, ignore_errors=True)
    return meta


def load_tables(outdir, names, mmap=True):
    """Carga las tablas guardadas (np.load mmap) con el mismo formato que _concat_tables."""
    tabs = {}
    for nm in names:
        d = os.path.join(outdir, nm)
        tabs[nm] = {k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r" if mmap and k in ("ckey", "cstart", "psym", "pcnt") else None)
                    for k in ("ckey", "cstart", "psym", "pcnt", "c_off", "cs_off", "D")}
    return tabs

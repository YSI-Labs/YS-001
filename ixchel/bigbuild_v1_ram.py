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


def build_chains_big(model, train, cut, outdir, nparts=16, group=4, chunk=20_000_000, chains=None, t0=16, verbose=True):
    """model: Ixchel (para _atoms_for/_H/chain_idx). train: array/memmap uint8. Construye train[:cut]. Guarda en outdir:
    <chain>/{ckey,cstart,psym,pcnt,c_off,cs_off,D}.npy y meta.json. Devuelve dict de tiempos."""
    import math
    names = chains or model.names
    shift = 64 - int(math.log2(nparts))
    os.makedirs(outdir, exist_ok=True)
    tmp = os.path.join(outdir, "_parts"); os.makedirs(tmp, exist_ok=True)
    L1 = {nm: len(model.chain_idx[nm][1]) - 1 for nm in names}
    stats = dict(scan_s=0.0, reduce_s=0.0, assemble_s=0.0, scans=0)
    T0 = time.time()
    for g0 in range(0, nparts, group):
        t_g = time.time()
        raw = {nm: ([], []) for nm in names}                      # nivel superior: (listas de upk, listas de cnt)
        cont = {(nm, l): ([], []) for nm in names for l in range(L1[nm] - 1)}   # (pk hijo unicos, pk padre)
        lo_p, hi_p = np.uint64(g0), np.uint64(g0 + group)
        for c0 in range(0, cut, chunk):
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
                pk_prev = None
                for l in range(top, -1, -1):
                    if l == top:
                        pk = (H[l] & MASK) | xs
                        pp = pk >> np.uint64(shift)
                        sel = (pp >= lo_p) & (pp < hi_p)
                        if sel.any():
                            u, c = np.unique(pk[sel], return_counts=True)
                            raw[nm][0].append(u); raw[nm][1].append(c.astype(np.int64))
                        pk_child = pk
                    if l >= 1:
                        # pares hijo = nivel l ; padre = nivel l-1 ; la tabla de continuacion es del nivel l-1
                        if l != top:
                            pk_child = (H[l] & MASK) | xs
                        ppk = (H[l - 1] & MASK) | xs
                        pp2 = ppk >> np.uint64(shift)
                        sel2 = (pp2 >= lo_p) & (pp2 < hi_p)
                        if sel2.any():
                            u2, idx = np.unique(pk_child[sel2], return_index=True)
                            cont[(nm, l - 1)][0].append(u2); cont[(nm, l - 1)][1].append(ppk[sel2][idx])
                        pk_child = None
                del H
            del A
            stats["scans"] += 1
        stats["scan_s"] += time.time() - t_g
        t_r = time.time()
        for nm in names:
            Ul, Cl = raw[nm]
            if Ul:
                U, C = _reduce_counts(Ul, Cl)
            else:
                U, C = np.zeros(0, np.uint64), np.zeros(0, np.int64)
            for p, up, cp in _split_sorted(U, C, shift, g0, group):
                np.savez(os.path.join(tmp, f"{nm}_L{L1[nm]-1}_raw_{p}.npz"), u=up, c=cp)
            raw[nm] = None
            for l in range(L1[nm] - 1):
                PK, PP = cont[(nm, l)]
                if PK:
                    PKc = np.concatenate(PK); PPc = np.concatenate(PP)
                    order = np.argsort(PKc, kind="stable"); PKc = PKc[order]; PPc = PPc[order]
                    first = np.concatenate(([True], PKc[1:] != PKc[:-1]))
                    PPc = PPc[first]                           # un padre por par hijo unico
                    PPc.sort()
                    brk = np.flatnonzero(PPc[1:] != PPc[:-1]) + 1
                    starts = np.concatenate(([0], brk))
                    U = PPc[starts]; C = np.diff(np.concatenate((starts, [len(PPc)]))).astype(np.int64)
                else:
                    U, C = np.zeros(0, np.uint64), np.zeros(0, np.int64)
                for p, up, cp in _split_sorted(U, C, shift, g0, group):
                    np.savez(os.path.join(tmp, f"{nm}_L{l}_cont_{p}.npz"), u=up, c=cp)
                cont[(nm, l)] = None
        stats["reduce_s"] += time.time() - t_r
        if verbose:
            print(f"[bigbuild] grupo {g0}..{g0+group-1} listo ({time.time()-t_g:.0f}s, acumulado {time.time()-T0:.0f}s)", flush=True)
    # ---- ensamblaje de tablas por cadena
    t_a = time.time()
    meta = {}
    for nm in names:
        tabs = []
        for l in range(L1[nm]):
            kind = "raw" if l == L1[nm] - 1 else "cont"
            Us, Cs = [], []
            for p in range(nparts):
                z = np.load(os.path.join(tmp, f"{nm}_L{l}_{kind}_{p}.npz"))
                Us.append(z["u"]); Cs.append(z["c"])
            U = np.concatenate(Us); C = np.concatenate(Cs)
            tabs.append(_make_table(U, C))
            del Us, Cs, U, C
        T = _concat_tables(tabs)
        d = os.path.join(outdir, nm); os.makedirs(d, exist_ok=True)
        for k, v in T.items():
            np.save(os.path.join(d, k + ".npy"), v)
        meta[nm] = dict(levels=L1[nm], pairs=int(len(T["psym"])), contexts=int(len(T["ckey"])))
        del tabs, T
        if verbose:
            print(f"[bigbuild] cadena {nm}: {meta[nm]}", flush=True)
    stats["assemble_s"] = time.time() - t_a
    stats["total_s"] = time.time() - T0
    meta["_stats"] = stats; meta["_cfg"] = dict(nparts=nparts, group=group, chunk=chunk, cut=int(cut), t0=t0)
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=1)
    return meta


def load_tables(outdir, names, mmap=True):
    """Carga las tablas guardadas (np.load mmap) con el mismo formato que _concat_tables."""
    tabs = {}
    for nm in names:
        d = os.path.join(outdir, nm)
        tabs[nm] = {k: np.load(os.path.join(d, k + ".npy"), mmap_mode="r" if mmap and k in ("ckey", "cstart", "psym", "pcnt") else None)
                    for k in ("ckey", "cstart", "psym", "pcnt", "c_off", "cs_off", "D")}
    return tabs

"""Verifica _atoms (numba, r4) contra una referencia en Python lineal y muestra valores sobre un snippet.
Uso: NUMBA_NUM_THREADS=2 python -I test_atoms_ref.py"""
import os, sys, json
os.environ["IXCHEL_DATA"] = "mix15"; os.environ.setdefault("NUMBA_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import core_codeatoms as C

ROOT = C.ROOT
itos = json.load(open(os.path.join(ROOT, "data", "mix15", "vocab.json")))
isl = C.build_char_tables(itos)[1]; isnl = C.build_char_tables(itos)[2]
isid, tok, kwh = C.build_code_tables(itos)
M = (1 << 64) - 1
P = 1099511628211
idx = {ch: i for i, ch in enumerate(itos)}


def ref_new(ids):
    """Referencia lineal: devuelve dict de listas (atomos 36..44) por posicion t (estado ANTES de consumir ids[t])."""
    out = {k: [] for k in ("ind", "indp", "bdep", "kwl", "fwp", "strst", "idl", "idp", "endc")}
    ind_sp = 0; lead = True; indp = 0; bdep = 0; kwl = 0; fst = 0; fh = 0; fwp = 0; st = 0
    q1 = q2 = 255; lastnsp = 255; endc = 0; inid = False; ih = 0; pid1 = 0; pid2 = 0
    for c in ids:
        c = int(c)
        out["ind"].append(min(ind_sp // 4, 15)); out["indp"].append(min(indp, 15)); out["bdep"].append(min(bdep, 15))
        out["kwl"].append(kwl); out["fwp"].append(fwp >> 2); out["strst"].append(st)
        out["idl"].append(pid1 >> 2); out["idp"].append(pid2 >> 2); out["endc"].append(endc)
        tc = int(tok[c]); s0 = st
        if isnl[c] == 1:
            if inid:
                pid2 = pid1; pid1 = ih; inid = False
                for j, h in enumerate(kwh.tolist()):
                    if h == pid1: kwl = j + 1
            ih = 0
            indp = ind_sp // 4
            fwp = fh if fst > 0 else 0
            tl = int(tok[lastnsp])
            endc = 1 if tl == 10 else (2 if tl == 4 else (3 if tl == 11 else 0))
            ind_sp = 0; lead = True; fst = 0; fh = 0; lastnsp = 255
            if st in (1, 2, 5): st = 0
        else:
            if s0 == 0:
                if isid[c] == 1:
                    ih = ((ih + c + 1) * P) & M; inid = True
                elif inid:
                    pid2 = pid1; pid1 = ih
                    for j, h in enumerate(kwh.tolist()):
                        if h == pid1: kwl = j + 1
                    inid = False; ih = 0
            isws = tc in (12, 13)
            if lead:
                if tc == 12: ind_sp += 1
                elif tc == 13: ind_sp += 4
                else: lead = False
            if not isws:
                lastnsp = c
                if fst == 0:
                    fh = ((fh + c + 1) * P) & M; fst = 1 if isid[c] == 1 else 2
                elif fst == 1:
                    if isid[c] == 1: fh = ((fh + c + 1) * P) & M
                    else: fst = 2
            if s0 == 0:
                if tc == 3: st = 5
                elif tc == 2: st = 3 if (q1 == 2 and q2 == 2) else 2
                elif tc == 1: st = 4 if (q1 == 1 and q2 == 1) else 1
                elif tc in (4, 6, 8): bdep += 1
                elif tc in (5, 7, 9):
                    if bdep > 0: bdep -= 1
            elif s0 == 1:
                if tc == 1: st = 0
            elif s0 == 2:
                if tc == 2: st = 0
            elif s0 == 3:
                if tc == 2 and q1 == 2 and q2 == 2: st = 0
            elif s0 == 4:
                if tc == 1 and q1 == 1 and q2 == 1: st = 0
        q2 = q1; q1 = tc
    return out


def numba_new(ids):
    cols = np.arange(45, dtype=np.int64)
    is64 = np.isin(cols, C.I64_ATOMS)
    dst = np.zeros(len(cols), dtype=np.int64); n8 = n64 = 0
    for i in range(len(cols)):
        if is64[i]: dst[i] = n64; n64 += 1
        else: dst[i] = n8; n8 += 1
    out8 = np.zeros((len(ids), n8), np.uint8); out64 = np.zeros((len(ids), n64), np.int64)
    C._atoms(ids, C.build_char_tables(itos)[0], isl, isnl, cols, is64, dst, out8, out64, isid, tok, kwh)
    res = {}
    for name in ("ind", "indp", "bdep", "kwl", "fwp", "strst", "idl", "idp", "endc"):
        a = C.ATOM_NAMES[name]
        if is64[a]: res[name] = out64[:, dst[a]]
        else: res[name] = out8[:, dst[a]].astype(np.int64)
    return res


if __name__ == "__main__":
    val = np.load(os.path.join(ROOT, "data", "mix15", "val_code.npy"))[:6000].astype(np.uint8)
    R = ref_new(val); N = numba_new(val)
    bad = {k: int(np.sum(np.array(R[k], dtype=np.int64) != N[k].astype(np.int64))) for k in R}
    print("mismatches (numba vs referencia) por atomo en 6000 chars de codigo:", bad)
    print("ok" if sum(bad.values()) == 0 else "FALLO")
    snippet = 'def f(x):\n    y = (x + 1,\n         2)\n    s = "a(b"  # c\n    """doc\n    """\n    return y\n'
    ids = np.array([idx[ch] for ch in snippet], dtype=np.uint8)
    S = numba_new(ids)
    print(f"{'char':>6} ind indp bdep kwl strst endc")
    for t in range(len(ids)):
        ch = snippet[t].replace("\n", "\\n")
        print(f"{ch:>6} {S['ind'][t]:>3} {S['indp'][t]:>4} {S['bdep'][t]:>4} {S['kwl'][t]:>3} {S['strst'][t]:>5} {S['endc'][t]:>4}")

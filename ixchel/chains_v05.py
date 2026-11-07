"""Cadenas de Ixchel v0.5: v0.1 (A,W,S,L) + lexicas (Kb,Kc,WL,W2) + cadenas de CODIGO podadas (r4_codeatoms: solo atomos bdep, fwp, idl)."""
from core_v05 import default_chains
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
CODE_FULL = dict(
    CK=[[], ["indp"], ["indp", "kwl"], ["indp", "kwl", "c1"]],
    CB=[[], ["bdep"], ["bdep", "ind"], ["bdep", "ind", "c1", "c2"]],
    CI=[[], ["idl"], ["idl", "c1"], ["idl", "idp", "c1"]],
    CF=[[], ["fwp"], ["fwp", "ind"], ["fwp", "ind", "wp"]],
    CS=[[], ["strst"], ["strst", "c1"], ["strst", "c1", "c2"], ["strst", "endc", "c1", "c2", "c3"]],
    CSK=[[], ["ind", "c1"], ["ind", "c1", "c3"], ["ind", "c1", "c3", "c4"]])
DROP = ("ind", "indp", "kwl", "idp", "strst", "endc")
def _drop(levels, atom):
    out = []
    for lvl in levels:
        l2 = [a for a in lvl if a != atom]
        if not out or out[-1] != l2: out.append(l2)
    return out
def code_chains_pruned():
    out = {}
    for k, lv in CODE_FULL.items():
        for a in DROP: lv = _drop(lv, a)
        if len(lv) > 1: out[k] = lv
    return out
def v05_chains():
    return {**default_chains(), **LEX, **code_chains_pruned()}

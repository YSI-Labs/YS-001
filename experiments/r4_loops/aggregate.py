"""r4_loops: tabla de resultados desde results/*.json y *_nll.npz.
 - Generacion: rep4, maxrep, ttr medios (generado vs referencia real) y diferencia PAREADA contra la base (mismos prompts), SE por texto.
 - Qwen: bpc de la continuacion (generado vs real, mismo prompt), diferencia pareada contra la base, SE por texto.
 - bpc real (eval_suite.domain_bpc, mix15): diferencia pareada contra 'orig' con SE por bloques de 500 posiciones.
Uso: python aggregate.py <base_label> <label> [<label> ...]"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")

def load(p):
    return json.load(open(os.path.join(R, p), encoding="utf-8")) if os.path.exists(os.path.join(R, p)) else None

def gen_table(base, labels):
    b = load(f"gen_{base}.json")
    key = lambda t: (t["domain"], t["idx"])
    bmap = {key(t): t for t in b["texts"]}
    print("| variante | T | rep4 gen | maxrep gen | ttr gen | rep4 real | maxrep real | ttr real | d rep4 vs base (SE) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lab in labels:
        g = load(f"gen_{lab}.json")
        if g is None: continue
        T = [t for t in g["texts"]]
        rg = np.array([[t["gen_m"]["rep4"], t["gen_m"]["maxrep"], t["gen_m"]["ttr"]] for t in T])
        rr = np.array([[t["ref_m"]["rep4"], t["ref_m"]["maxrep"], t["ref_m"]["ttr"]] for t in T])
        d = np.array([t["gen_m"]["rep4"] - bmap[key(t)]["gen_m"]["rep4"] for t in T if key(t) in bmap])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"| {lab} ({g['recipe']}{'' if g['top_p'] is None and g['penalty'] is None else ' +muestreo etiquetado'}) | {g['T']} | "
              f"{rg[:,0].mean():.3f} | {rg[:,1].mean():.1f} | {rg[:,2].mean():.3f} | {rr[:,0].mean():.3f} | {rr[:,1].mean():.1f} | {rr[:,2].mean():.3f} | "
              f"{d.mean():+.3f} ({se:.3f}) |")

def qwen_table(base, labels):
    b = load(f"qwen_{base}.json")
    bm = {(t["domain"], t["idx"]): t["gen_bpc"] for t in b["texts"]}
    print("| variante | T | n | bpc Qwen gen | bpc Qwen real (ref) | ppl gen (2^bpc) | gen-real | gen-base pareado (SE) |")
    print("|---|---|---|---|---|---|---|---|")
    for lab in labels:
        q = load(f"qwen_{lab}.json")
        if q is None: continue
        gb = np.array([t["gen_bpc"] for t in q["texts"]]); rb = np.array([t["ref_bpc"] for t in q["texts"]])
        d = np.array([t["gen_bpc"] - bm[(t["domain"], t["idx"])] for t in q["texts"]])
        se = d.std(ddof=1) / np.sqrt(len(d))
        g = load(f"gen_{lab}.json")
        print(f"| {lab} | {g['T'] if g else '?'} | {len(gb)} | {gb.mean():.4f} | {rb.mean():.4f} | {2**gb.mean():.3f} | {gb.mean()-rb.mean():+.4f} | {d.mean():+.4f} ({se:.4f}) |")

def bpc_table(labels):
    o = load("bpc_orig.json")
    if o is None: return
    no = np.load(os.path.join(R, "bpc_orig_nll.npz"))
    print("| receta | web | code | math | chat | books | diferencia pareada vs base (SE por bloques de 500) por dominio |")
    print("|---|---|---|---|---|---|---|")
    for lab in labels:
        f = load(f"bpc_{lab}.json")
        if f is None: continue
        nl = np.load(os.path.join(R, f"bpc_{lab}_nll.npz"))
        cells, diffs = [], []
        for d in ("web", "code", "math", "chat", "books"):
            cells.append(f"{f['domain_bpc'][d]['bpc']:.4f}")
            x = nl[d] - no[d]; nb = len(x) // 500
            bm = x[:nb * 500].reshape(nb, 500).mean(1)
            diffs.append(f"{d} {x.mean():+.4f} ({bm.std(ddof=1)/np.sqrt(nb):.4f})")
        print(f"| {lab} | " + " | ".join(cells) + " | " + "; ".join(diffs) + " |")

if __name__ == "__main__":
    base = sys.argv[1]; labels = sys.argv[2:]
    print("## Generacion (metricas automaticas, 90 textos por variante)")
    gen_table(base, [base] + labels)
    print("\n## Qwen2.5-1.5B (30 textos por variante, continuacion condicionada al prompt)")
    qwen_table(base, [base] + labels)
    print("\n## bpc sobre texto real (mix15 val, 30k posiciones/dominio)")
    bpc_table(["orig"] + labels)

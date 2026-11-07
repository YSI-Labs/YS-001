"""r4_loops: genera textos con una receta (o con muestreo etiquetado) desde prompts de val de web/code/chat.

Uso: python gen_run.py <pickle_base> <label> <recipe> <T> [top_p|-] [penalty|-] [n_per_domain] [n_chars]
Salida: results/gen_<label>.json con, por texto: dominio, prompt, generado, referencia real (mismos prompts), y metricas
de repeticion (rep4 = fraccion de 4-gramas de palabras repetidos, maxrep = mayor subcadena repetida en chars,
ttr = tipos distintos / palabras) para generado y referencia.
Prompts: 200 chars de val_<dom> de mix1 (inicio en salto de linea). Continuacion: 400 chars. Semilla fija por texto."""
import os, sys, json, pickle, time
os.environ["NUMBA_NUM_THREADS"] = "2"; os.environ["OMP_NUM_THREADS"] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import gen_lib as G
from gen_lib import words

ROOT = __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
DATA1 = os.path.join(ROOT, "data", "mix1")
PROMPT, NCH = 200, 400
DOMAINS = ("web", "code", "chat")

def maxrep(s):
    n = len(s); best = 0
    prev = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        cur = [0] * (n + 1)
        for j in range(n - 1, i, -1):
            if s[i] == s[j]:
                v = prev[j + 1] + 1
                cur[j] = v
                if v > best:
                    best = v
        prev = cur
    return best

def metrics(s):
    w = words(s)
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    rep4 = 1.0 - len(set(g)) / len(g) if g else 0.0
    ttr = len(set(w)) / len(w) if w else 0.0
    return dict(rep4=rep4, maxrep=maxrep(s), ttr=ttr, nw=len(w))

def prompt_starts(val, n, margin):
    text_len = len(val)
    starts = np.linspace(0, text_len - PROMPT - NCH - margin, n).astype(int)
    return starts

def main():
    pkl, label, recipe, T = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    top_p = None if len(sys.argv) <= 5 or sys.argv[5] == "-" else float(sys.argv[5])
    penalty = None if len(sys.argv) <= 6 or sys.argv[6] == "-" else float(sys.argv[6])
    npd = int(sys.argv[7]) if len(sys.argv) > 7 else 30
    nch = int(sys.argv[8]) if len(sys.argv) > 8 else NCH
    itos = __import__("json").load(open(os.path.join(ROOT, "data", "mix15", "vocab.json")))
    t0 = time.time()
    model = pickle.load(open(pkl, "rb")); model.itos = itos
    print("load s", round(time.time() - t0, 1), flush=True)
    out = []
    t1 = time.time()
    nlid = itos.index("\n")
    for di, d in enumerate(DOMAINS):
        val = np.load(os.path.join(DATA1, f"val_{d}.npy"))
        for k, s0 in enumerate(prompt_starts(val, npd, 10)):
            s = int(s0)
            hit = np.flatnonzero(val[s:s + 200] == nlid)          # inicio de la linea siguiente
            s += (int(hit[0]) + 1) if len(hit) else 0
            pr = val[s:s + PROMPT]; ref = val[s + PROMPT:s + PROMPT + nch]
            rng = np.random.default_rng(1000 * (k + 1) + di)
            gen = G.generate(model, pr, nch, recipe, rng, T=T, top_p=top_p, penalty=penalty)
            gs = "".join(itos[i] for i in gen); rs = "".join(itos[i] for i in ref); ps = "".join(itos[i] for i in pr)
            out.append(dict(domain=d, idx=k, prompt=ps, gen=gs, ref=rs, gen_m=metrics(gs), ref_m=metrics(rs)))
            print(d, k, round(time.time() - t1), "s", flush=True)
    res = dict(label=label, recipe=recipe, T=T, top_p=top_p, penalty=penalty, n_per_domain=npd, n_chars=nch,
               texts=out, gen_seconds=round(time.time() - t1, 1))
    json.dump(res, open(os.path.join(HERE, "results", f"gen_{label}.json"), "w"), ensure_ascii=False, indent=1)
    print("saved", label, flush=True)

if __name__ == "__main__":
    main()

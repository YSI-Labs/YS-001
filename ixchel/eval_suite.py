"""Bateria de evaluacion para modelos de caracteres sobre el corpus mezclado mix1 (IXCHEL_DATA=mix1).
Funciones puras que reciben un modelo con predict(ids,pos)->[m,V] (contrato del harness) y 'enc' (str -> ids uint8).
 - bpc por dominio (documentos retenidos; codigo: paquetes retenidos)
 - Opcion multiple por log-verosimilitud (HellaSwag, PIQA, ARC-Easy) con acc y acc_norm (por caracter), azar=25/50/25 %
 - HumanEval (164 problemas) pass@1 con decodificacion voraz, ejecutado en subproceso aislado con timeout y filtro de llamadas peligrosas
 - Validez sintactica de codigo generado (ast.parse) a partir de prefijos de codigo retenido
"""
import os, sys, json, re, ast, gzip, subprocess, tempfile, textwrap, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw_mix")
LN2 = np.log(2.0)

# ---------------------------------------------------------------- utilidades
def make_encoder(itos):
    import numpy as np
    lut = np.full(256, itos.index("?"), dtype=np.uint8)
    for i, c in enumerate(itos):
        lut[ord(c)] = i
    lut[ord("\r")] = itos.index("\n")
    UNI = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-", "…": "...", " ": " ",
           "•": "*", "é": "e", "è": "e", "à": "a", "ü": "u", "ö": "o", "ä": "a", "°": " deg"}
    def enc(s):
        for k, v in UNI.items():
            s = s.replace(k, v)
        return lut[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
    return enc

def loglik_batch(model, ctx, conts, enc, V):
    """log2-verosimilitud total de cada continuacion dada la misma 'ctx' (str). Devuelve (ll[len(conts)], nchars[len(conts)])."""
    c = enc(ctx); out, ns = [], []
    for cont in conts:
        a = enc(cont)
        ids = np.concatenate([c, a]).astype(np.uint8)
        if hasattr(model, "score_sequence"):          # redes recurrentes: una sola pasada por secuencia
            ll = model.score_sequence(ids, len(c))
        else:
            pos = np.arange(len(c), len(ids), dtype=np.int64)
            P = model.predict(ids, pos).astype(np.float64)
            ll = np.log2(np.maximum(P[np.arange(len(pos)), ids[pos]], 1e-12)).sum()
        out.append(ll); ns.append(len(a))
    return np.array(out), np.array(ns)

def acc_ci(correct):
    c = np.asarray(correct, dtype=float); n = len(c); p = c.mean()
    return dict(acc=float(p), se=float(np.sqrt(p * (1 - p) / n)), n=int(n))

# ---------------------------------------------------------------- bpc por dominio
def domain_bpc(model, data_dir, V, domains=("web", "code", "math", "chat", "books"), n_eval=30_000, block=500, batch=4096):
    import harness
    res = {}
    for d in domains:
        val = np.load(os.path.join(data_dir, f"val_{d}.npy"))
        pos = harness.eval_positions(val, min(n_eval, len(val) - 100))
        nll = np.zeros(len(pos)); top1 = np.zeros(len(pos))
        for s in range(0, len(pos), batch):
            pp = pos[s:s + batch]
            P = model.predict(val, pp).astype(np.float64)
            nll[s:s + len(pp)] = -np.log2(np.maximum(P[np.arange(len(pp)), val[pp]], 1e-12))
            top1[s:s + len(pp)] = (P.argmax(1) == val[pp])
        nb = len(nll) // block; bm = nll[:nb * block].reshape(nb, block).mean(1)
        res[d] = dict(bpc=float(nll.mean()), se_iid=float(nll.std(ddof=1) / np.sqrt(len(nll))), se_block=float(bm.std(ddof=1) / np.sqrt(nb)),
                      n=len(nll), val_chars=int(len(val)), top1_acc=float(top1.mean()))
        res[d]["nll"] = nll
    return res

# ---------------------------------------------------------------- opcion multiple
def _hs_clean(t):
    t = t.strip().replace(" [title]", ". "); t = re.sub(r"\[.*?\]", "", t); return t.replace("  ", " ")

def hellaswag(model, enc, V, n=5000, seed=0):
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, "hellaswag", "validation.parquet"))
    idx = np.random.default_rng(seed).permutation(len(df))[:n]
    cor, cor_n = [], []
    for i in idx:
        r = df.iloc[i]
        ctx = _hs_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        ends = [" " + _hs_clean(e) for e in r["endings"]]
        ll, ns = loglik_batch(model, ctx, ends, enc, V)
        y = int(r["label"]); cor.append(int(np.argmax(ll)) == y); cor_n.append(int(np.argmax(ll / ns)) == y)
    return dict(acc=acc_ci(cor), acc_norm=acc_ci(cor_n), random=0.25)

def piqa(model, enc, V, n=1838):
    rows = [json.loads(l) for l in open(os.path.join(RAW, "piqa", "valid.jsonl"), encoding="utf-8")]
    labs = [int(x) for x in open(os.path.join(RAW, "piqa", "valid-labels.lst")).read().split()]
    cor, cor_n = [], []
    for r, y in list(zip(rows, labs))[:n]:
        ctx = f"Question: {r['goal']}\nAnswer:"
        ll, ns = loglik_batch(model, ctx, [" " + r["sol1"], " " + r["sol2"]], enc, V)
        cor.append(int(np.argmax(ll)) == y); cor_n.append(int(np.argmax(ll / ns)) == y)
    return dict(acc=acc_ci(cor), acc_norm=acc_ci(cor_n), random=0.5)

def arc_easy(model, enc, V, n=2376):
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, "arc", "easy_test.parquet"))
    cor, cor_n = [], []
    for _, r in df.head(n).iterrows():
        labels = list(r["choices"]["label"]); texts = list(r["choices"]["text"])
        if r["answerKey"] not in labels: continue
        y = labels.index(r["answerKey"])
        ll, ns = loglik_batch(model, f"Question: {r['question']}\nAnswer:", [" " + t for t in texts], enc, V)
        cor.append(int(np.argmax(ll)) == y); cor_n.append(int(np.argmax(ll / ns)) == y)
    return dict(acc=acc_ci(cor), acc_norm=acc_ci(cor_n), random=0.25)

# ---------------------------------------------------------------- generacion
def generate_greedy(model, enc, V, prompt, max_chars, itos, stops=(), temperature=0.0, rng=None, keep_ctx=None):
    ids = list(enc(prompt)); out = []
    buf = np.zeros(len(ids) + max_chars + 1, dtype=np.uint8); buf[:len(ids)] = ids; L = len(ids)
    for _ in range(max_chars):
        lo = 0 if keep_ctx is None else max(0, L - keep_ctx)
        p = model.predict(buf[lo:L + 1], np.array([L - lo], dtype=np.int64))[0].astype(np.float64)
        if temperature == 0.0:
            c = int(np.argmax(p))
        else:
            q = p ** (1.0 / temperature); q /= q.sum(); c = int(rng.choice(V, p=q))
        buf[L] = c; L += 1; out.append(itos[c])
        s = "".join(out)
        if any(st in s for st in stops):
            break
    s = "".join(out)
    cut = len(s)
    for st in stops:
        j = s.find(st)
        if j != -1: cut = min(cut, j)
    return s[:cut]

DANGER = ("import os", "import sys", "subprocess", "shutil", "open(", "remove", "rmdir", "system", "__import__", "eval(", "exec(",
          "socket", "urllib", "requests", "pathlib", "unlink", "rmtree", "getattr", "globals", "input(")

def run_candidate(code, timeout=6):
    if any(d in code for d in DANGER):
        return False, "rechazado por filtro de seguridad"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cand.py"); open(p, "w", encoding="utf-8").write(code)
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=td, capture_output=True, timeout=timeout, text=True)
            return r.returncode == 0, (r.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"

def humaneval(model, enc, V, itos, max_chars=350, limit=None, keep_ctx=1500):
    probs = [json.loads(l) for l in gzip.open(os.path.join(RAW, "humaneval", "HumanEval.jsonl.gz"), "rt", encoding="utf-8")]
    if limit: probs = probs[:limit]
    stops = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n\n\n")
    ok, syn, samples, nonblank, syn_nb = 0, 0, [], 0, 0
    for pr in probs:
        comp = generate_greedy(model, enc, V, pr["prompt"], max_chars, itos, stops=stops, keep_ctx=keep_ctx)
        prog = pr["prompt"] + comp + "\n\n" + pr["test"] + f"\n\ncheck({pr['entry_point']})\n"
        nb_ok = len(re.sub(r"\s", "", comp)) >= 10      # completacion NO trivial (>=10 caracteres no blancos)
        nonblank += int(nb_ok)
        try:
            ast.parse(pr["prompt"] + comp + "\n    pass\n"); syn += 1; syn_nb += int(nb_ok)
        except Exception:
            try:
                ast.parse(pr["prompt"] + comp); syn += 1; syn_nb += int(nb_ok)
            except Exception:
                pass
        passed, msg = run_candidate(prog)
        ok += int(passed)
        if len(samples) < 3: samples.append(dict(task=pr["task_id"], completion=comp[:300], passed=bool(passed)))
    n = len(probs)
    return dict(pass_at_1=ok / n, n=n, syntax_valid_frac=syn / n, nonblank_frac=nonblank / n, syntax_valid_among_nonblank=syn_nb / n, samples=samples)

def code_syntax_validity(model, enc, V, itos, data_dir, n_prompts=120, prefix_chars=600, gen_chars=250, seed=0, keep_ctx=1500):
    """Prefijos de CODIGO retenido (paquetes no vistos) cortados en limite de linea; el modelo continua; mide la fraccion de
    (prefijo + continuacion truncada a la ultima linea completa) que sigue siendo sintacticamente valida como fragmento."""
    val = np.load(os.path.join(data_dir, "val_code.npy")); text = "".join(itos[i] for i in val)
    rng = np.random.default_rng(seed); starts = rng.integers(0, len(text) - prefix_chars - gen_chars, size=n_prompts)
    ok_gen = ok_ref = 0
    for s in starts:
        s = text.find("\n", s) + 1
        prefix = text[s:s + prefix_chars]; prefix = prefix[:prefix.rfind("\n") + 1]
        ref = text[s + len(prefix):s + len(prefix) + gen_chars]
        gen = generate_greedy(model, enc, V, prefix, gen_chars, itos, keep_ctx=keep_ctx, temperature=0.8, rng=rng)
        def valid(frag):
            frag = frag[:frag.rfind("\n") + 1] if "\n" in frag else frag
            for tail in ("", "    pass\n", "pass\n"):
                try: ast.parse(textwrap.dedent(prefix + frag + tail)); return True
                except Exception: pass
            try: ast.parse(textwrap.dedent(frag)); return True
            except Exception: return False
        ok_gen += int(valid(gen)); ok_ref += int(valid(ref))
    return dict(gen_valid=ok_gen / n_prompts, real_continuation_valid=ok_ref / n_prompts, n=n_prompts)

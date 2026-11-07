"""Evalua un modelo de lenguaje REAL (transformer abierto, solo como VARA DE CALIBRACION) con la MISMA bateria que Ixchel:
bpc por dominio (mismo texto retenido de mix1; bits totales / caracteres => comparable con cualquier tokenizador), HellaSwag/PIQA/ARC-Easy por
log-verosimilitud con acc_norm por caracteres, y HumanEval pass@1 voraz con ejecucion aislada (eval_suite.run_candidate).
Uso: python -I eval_ref_model.py <dir_modelo> <tag> [quick] [threads]"""
import os, sys, json, re, time, ast, gzip
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, torch
import eval_suite as ES
from transformers import AutoModelForCausalLM, AutoTokenizer

D = os.path.join(ROOT, "data", "mix1"); RAW = os.path.join(ROOT, "data", "raw_mix")
LN2 = float(np.log(2.0))

def load(path, threads):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32); model.eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return tok, model, start

@torch.no_grad()
def text_nll_bits(tok, model, start, text):
    ids = [start] + tok(text, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids]); lg = model(x).logits[0, :-1].float()
    lp = torch.log_softmax(lg, -1); tgt = torch.tensor(ids[1:])
    return float(-(lp[torch.arange(len(tgt)), tgt]).sum() / LN2), len(ids) - 1

def domain_bpc(tok, model, start, itos, n_seg=24, seg_chars=1500):
    res = {}
    for d in ("web", "code", "math", "chat", "books"):
        val = np.load(os.path.join(D, f"val_{d}.npy")); text = "".join(itos[i] for i in val)
        starts = np.linspace(0, len(text) - seg_chars - 2, n_seg).astype(int)
        bits = chars = ntok = 0.0
        for s in starts:
            s = text.find("\n", s) + 1; seg = text[s:s + seg_chars]
            b, nt = text_nll_bits(tok, model, start, seg); bits += b; chars += len(seg); ntok += nt
        res[d] = dict(bpc=bits / chars, chars=int(chars), tokens=int(ntok))
    return res

@torch.no_grad()
def loglik_choices(tok, model, start, ctx, conts):
    """Devuelve (ll_bits[len(conts)], nchars[len(conts)]) con tokenizacion conjunta ctx+cont."""
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    seqs, spans = [], []
    for c in conts:
        whole = tok(ctx + c, add_special_tokens=False)["input_ids"]
        if whole[:len(ctx_ids)] == ctx_ids:
            ids = [start] + whole; lo = 1 + len(ctx_ids)
        else:
            cid = tok(c, add_special_tokens=False)["input_ids"]; ids = [start] + ctx_ids + cid; lo = 1 + len(ctx_ids)
        seqs.append(ids); spans.append((lo, len(ids)))
    L = max(len(s) for s in seqs); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    X = torch.full((len(seqs), L), pad, dtype=torch.long); M = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs): X[i, :len(s)] = torch.tensor(s); M[i, :len(s)] = 1
    lp = torch.log_softmax(model(input_ids=X, attention_mask=M).logits[:, :-1].float(), -1)
    out = []
    for i, (lo, hi) in enumerate(spans):
        tgt = X[i, lo:hi]; out.append(float(lp[i, torch.arange(lo - 1, hi - 1), tgt].sum() / LN2))
    return np.array(out), np.array([len(c) for c in conts])

def mc_hellaswag(tok, model, start, n):
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, "hellaswag", "validation.parquet")); idx = np.random.default_rng(0).permutation(len(df))[:n]
    cor, cor_n = [], []
    for i in idx:
        r = df.iloc[i]; ctx = ES._hs_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        ll, ns = loglik_choices(tok, model, start, ctx, [" " + ES._hs_clean(e) for e in r["endings"]]); y = int(r["label"])
        cor.append(int(np.argmax(ll)) == y); cor_n.append(int(np.argmax(ll / ns)) == y)
    return dict(acc=ES.acc_ci(cor), acc_norm=ES.acc_ci(cor_n), random=0.25)

def mc_piqa(tok, model, start, n):
    rows = [json.loads(l) for l in open(os.path.join(RAW, "piqa", "valid.jsonl"), encoding="utf-8")]
    labs = [int(x) for x in open(os.path.join(RAW, "piqa", "valid-labels.lst")).read().split()]
    cor, cor_n = [], []
    for r, y in list(zip(rows, labs))[:n]:
        ll, ns = loglik_choices(tok, model, start, f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]])
        cor.append(int(np.argmax(ll)) == y); cor_n.append(int(np.argmax(ll / ns)) == y)
    return dict(acc=ES.acc_ci(cor), acc_norm=ES.acc_ci(cor_n), random=0.5)

def mc_arc(tok, model, start, n):
    import pandas as pd
    df = pd.read_parquet(os.path.join(RAW, "arc", "easy_test.parquet")); cor, cor_n = [], []
    for _, r in df.head(n).iterrows():
        labels = list(r["choices"]["label"]); texts = list(r["choices"]["text"])
        if r["answerKey"] not in labels: continue
        y = labels.index(r["answerKey"])
        ll, ns = loglik_choices(tok, model, start, f"Question: {r['question']}\nAnswer:", [" " + t for t in texts])
        cor.append(int(np.argmax(ll)) == y); cor_n.append(int(np.argmax(ll / ns)) == y)
    return dict(acc=ES.acc_ci(cor), acc_norm=ES.acc_ci(cor_n), random=0.25)

@torch.no_grad()
def humaneval(tok, model, limit=None, max_new=256):
    probs = [json.loads(l) for l in gzip.open(os.path.join(RAW, "humaneval", "HumanEval.jsonl.gz"), "rt", encoding="utf-8")]
    if limit: probs = probs[:limit]
    stops = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n\n\n"]
    ok = rejected = 0; ntok = 0; t_gen = 0.0; samples = []
    for pr in probs:
        x = tok(pr["prompt"], return_tensors="pt"); t = time.time()
        out = model.generate(**x, max_new_tokens=max_new, do_sample=False, stop_strings=stops, tokenizer=tok, pad_token_id=tok.eos_token_id)
        t_gen += time.time() - t; new = out[0, x["input_ids"].shape[1]:]; ntok += len(new)
        comp = tok.decode(new, skip_special_tokens=True); cut = len(comp)
        for st in stops:
            j = comp.find(st)
            if j != -1: cut = min(cut, j)
        comp = comp[:cut]
        prog = pr["prompt"] + comp + "\n\n" + pr["test"] + f"\n\ncheck({pr['entry_point']})\n"
        passed, msg = ES.run_candidate(prog); ok += int(passed); rejected += int("filtro" in msg)
        if len(samples) < 3: samples.append(dict(task=pr["task_id"], completion=comp[:300], passed=bool(passed)))
    return dict(pass_at_1=ok / len(probs), n=len(probs), rejected_by_safety_filter=rejected, gen_tokens=int(ntok), gen_tok_per_s=ntok / max(t_gen, 1e-9), samples=samples)

if __name__ == "__main__":
    path, tag = sys.argv[1], sys.argv[2]; quick = "quick" in sys.argv[3:]
    threads = int([a for a in sys.argv[3:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[3:]) else 8
    itos = json.load(open(os.path.join(D, "vocab.json")))
    t0 = time.time(); tok, model, start = load(path, threads); out = dict(tag=tag, model=os.path.basename(path), threads=threads, load_s=time.time() - t0,
                                                                        n_params=int(sum(p.numel() for p in model.parameters())))
    resf = os.path.join(HERE, "results", f"ref_{tag}.json")
    def save(): json.dump(out, open(resf, "w"), indent=1)
    t = time.time(); out["domain_bpc"] = domain_bpc(tok, model, start, itos, n_seg=(4 if quick else 24)); out["domain_s"] = time.time() - t
    print("bpc", {d: round(v["bpc"], 3) for d, v in out["domain_bpc"].items()}, f"({out['domain_s']:.0f}s)", flush=True); save()
    for nm, fn, n in (("hellaswag", mc_hellaswag, 40 if quick else 500), ("piqa", mc_piqa, 40 if quick else 400), ("arc_easy", mc_arc, 40 if quick else 400)):
        t = time.time(); out[nm] = fn(tok, model, start, n); out[nm]["seconds"] = time.time() - t; print(nm, out[nm], flush=True); save()
    t = time.time(); out["humaneval"] = humaneval(tok, model, limit=(4 if quick else None)); out["humaneval"]["seconds"] = time.time() - t
    print("humaneval", {k: v for k, v in out["humaneval"].items() if k != "samples"}, flush=True); save(); print("REF_DONE", flush=True)

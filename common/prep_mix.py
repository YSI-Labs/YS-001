"""Corpus mezclado 'mix1' (estilo de datos de modelos de frontera, pero pequeno): web (C4), codigo (Python local), matematicas
(GSM8K), instrucciones (Dolly), libros (Shakespeare). Todo plegado a ASCII imprimible + \\n + \\t (V=97); lo no-ASCII -> '?'.
Particion POR DOCUMENTO; codigo: val = paquetes retenidos (scipy, networkx) para medir generalizacion a bibliotecas no vistas.
Salida: data/mix1/{train.npy, val.npy (concat equilibrada), val_<dominio>.npy, vocab.json, manifest.json}.
Uso: python -I prep_mix.py"""
import os, sys, json, gzip, zlib, hashlib, random, re, site
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw_mix"); OUT = os.path.join(ROOT, "data", "mix1"); os.makedirs(OUT, exist_ok=True)
rnd = random.Random(1234)

# ---------------- vocabulario ASCII
itos = ["\t", "\n"] + [chr(c) for c in range(32, 127)]
stoi = {c: i for i, c in enumerate(itos)}
lut = np.full(256, stoi["?"], dtype=np.uint8)
for c, i in stoi.items():
    lut[ord(c)] = i
lut[ord("\r")] = stoi["\n"]
UNI = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-", "…": "...", " ": " ",
       "•": "*", "é": "e", "è": "e", "à": "a", "ü": "u", "ö": "o", "ä": "a", "°": " deg"}
def fold(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    b = s.encode("ascii", "replace")           # no-ASCII -> b'?'
    return lut[np.frombuffer(b, dtype=np.uint8)]
def enc(s):
    return fold(s)

man = {}
domains_train = {}     # dominio -> lista de arrays (documentos)
val = {}               # dominio -> array

# ---------------- web: C4 (primeros ~120MB gz)
def read_c4(path, max_docs=10**9):
    d = zlib.decompressobj(wbits=31); buf = b""; docs = []
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk: break
            try: buf += d.decompress(chunk)
            except Exception: break
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                try: docs.append(json.loads(ln)["text"])
                except Exception: pass
    return docs
c4 = read_c4(os.path.join(RAW, "c4", "c4_part.json.gz"))
rnd.shuffle(c4)
web_val, web_tr, nv = [], [], 0
for t in c4:
    a = enc(t + "\n\n")
    if nv < 1_000_000: web_val.append(a); nv += len(a)
    else: web_tr.append(a)
TRAIN_WEB = 45_000_000
acc, keep = 0, []
for a in web_tr:
    if acc >= TRAIN_WEB: break
    keep.append(a); acc += len(a)
domains_train["web"] = keep; val["web"] = np.concatenate(web_val)
man["web"] = dict(source="allenai/c4 en shard 0 (primeros 120MB gz), documentos mezclados", train_docs=len(keep), train_chars=acc, val_chars=len(val["web"]))

# ---------------- codigo: Python local
roots = [os.path.join(sys.base_prefix, "Lib")]
sp = os.path.join(sys.base_prefix, "Lib", "site-packages")
HELD = {"scipy", "networkx"}
files_tr, files_va = [], []
seen = set()
for dp, dn, fn in os.walk(roots[0]):
    rel = os.path.relpath(dp, roots[0]).replace("\\", "/")
    in_sp = rel.startswith("site-packages")
    if "__pycache__" in dp: continue
    top = rel.split("/")[1] if in_sp and "/" in rel else ""
    for f in fn:
        if not f.endswith(".py"): continue
        p = os.path.join(dp, f)
        try:
            if os.path.getsize(p) > 400_000 or os.path.getsize(p) < 200: continue
        except Exception: continue
        (files_va if (in_sp and top in HELD) else files_tr).append(p)
def load_files(paths, cap_chars, per_file=60_000):
    rnd.shuffle(paths); out, tot = [], 0
    for p in paths:
        if tot >= cap_chars: break
        try: s = open(p, encoding="utf-8", errors="replace").read()
        except Exception: continue
        h = hashlib.md5(s.encode("utf-8", "replace")).hexdigest()
        if h in seen: continue
        seen.add(h)
        a = enc(s[:per_file] + "\n\n"); out.append(a); tot += len(a)
    return out, tot
code_tr, nct = load_files(files_tr, 20_000_000)
code_va, ncv = load_files(files_va, 1_000_000)
domains_train["code"] = code_tr; val["code"] = np.concatenate(code_va)
man["code"] = dict(source="Python stdlib + site-packages locales (sin scipy/networkx), dedup md5, <=60k chars/archivo; val = scipy+networkx (paquetes retenidos)",
                   train_docs=len(code_tr), train_chars=nct, val_docs=len(code_va), val_chars=ncv)

# ---------------- matematicas: GSM8K
def gsm(path):
    out = []
    for ln in open(path, encoding="utf-8"):
        j = json.loads(ln); out.append(f"Question: {j['question']}\nAnswer: {j['answer']}\n\n")
    return out
g_tr = [enc(t) for t in gsm(os.path.join(RAW, "gsm8k", "train.jsonl"))]
g_va = [enc(t) for t in gsm(os.path.join(RAW, "gsm8k", "test.jsonl"))]
domains_train["math"] = g_tr; val["math"] = np.concatenate(g_va)
man["math"] = dict(source="openai/grade-school-math train/test", train_docs=len(g_tr), train_chars=int(sum(map(len, g_tr))), val_chars=len(val["math"]))

# ---------------- instrucciones: Dolly
dl = []
for ln in open(os.path.join(RAW, "dolly", "dolly.jsonl"), encoding="utf-8"):
    j = json.loads(ln)
    s = f"### Instruction:\n{j['instruction']}\n\n"
    if j.get("context"): s += f"### Context:\n{j['context']}\n\n"
    s += f"### Response:\n{j['response']}\n\n"
    dl.append(s)
rnd.shuffle(dl)
nva = max(1, int(0.05 * len(dl)))
d_va = [enc(t) for t in dl[:nva]]; d_tr = [enc(t) for t in dl[nva:]]
domains_train["chat"] = d_tr; val["chat"] = np.concatenate(d_va)
man["chat"] = dict(source="databricks-dolly-15k", train_docs=len(d_tr), train_chars=int(sum(map(len, d_tr))), val_chars=len(val["chat"]))

# ---------------- libros: Shakespeare (texto limpio de Gutenberg #100)
sh = open(os.path.join(ROOT, "data", "shakespeare", "clean.txt"), encoding="utf-8").read()
ids = enc(sh); cut = int(len(ids) * 0.9)
# partir en documentos de ~20k chars para poder barajar
tr_sh = ids[:cut]; docs = [tr_sh[i:i + 20_000] for i in range(0, len(tr_sh), 20_000)]
domains_train["books"] = docs; val["books"] = ids[cut:]
man["books"] = dict(source="Project Gutenberg #100 (Shakespeare), ultimo 10% como val", train_docs=len(docs), train_chars=int(cut), val_chars=len(val["books"]))

# ---------------- mezcla de entrenamiento (barajada por documento)
all_docs = [(d, a) for d, lst in domains_train.items() for a in lst]
rnd.shuffle(all_docs)
train = np.concatenate([a for _, a in all_docs])
np.save(os.path.join(OUT, "train.npy"), train)
VAL_CAP = 500_000
vcat = []
for d in ("web", "code", "math", "chat", "books"):
    np.save(os.path.join(OUT, f"val_{d}.npy"), val[d]); vcat.append(val[d][:VAL_CAP])
np.save(os.path.join(OUT, "val.npy"), np.concatenate(vcat))
json.dump(itos, open(os.path.join(OUT, "vocab.json"), "w"))
man["train_total_chars"] = int(len(train)); man["V"] = len(itos)
man["train_share"] = {d: round(man[d]["train_chars"] / len(train), 4) for d in domains_train}
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(json.dumps(man, indent=1))

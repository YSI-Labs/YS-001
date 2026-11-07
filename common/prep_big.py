"""Corpus GRANDE 'big1' (~700M caracteres) con la misma receta y vocabulario que mix1: web (C4 shards 1-3), codigo (Python local sin scipy/networkx),
dolly, gsm8k, Shakespeare. Los val_* son los MISMOS que mix1 (web = shard 0, codigo = scipy+networkx), asi que no hay solapamiento.
Uso: python -I prep_big.py [web_chars] [code_chars]"""
import os, sys, json, gzip, hashlib, random, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw_mix"); MIX = os.path.join(ROOT, "data", "mix1"); OUT = os.path.join(ROOT, "data", "big1"); os.makedirs(OUT, exist_ok=True)
WEB = int(float(sys.argv[1])) if len(sys.argv) > 1 else 480_000_000
CODE = int(float(sys.argv[2])) if len(sys.argv) > 2 else 190_000_000
rnd = random.Random(777)
itos = json.load(open(os.path.join(MIX, "vocab.json"))); stoi = {c: i for i, c in enumerate(itos)}
lut = np.full(256, stoi["?"], dtype=np.uint8)
for c, i in stoi.items(): lut[ord(c)] = i
lut[ord("\r")] = stoi["\n"]
UNI = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-", "…": "...", " ": " ", "•": "*",
       "é": "e", "è": "e", "à": "a", "ü": "u", "ö": "o", "ä": "a", "°": " deg"}
def enc(s):
    for k, v in UNI.items(): s = s.replace(k, v)
    return lut[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
docs = []; man = {}
# web
web, nweb = [], 0
for i in ("00001", "00002", "00003"):
    with gzip.open(os.path.join(RAW, "c4big", f"c4_{i}.json.gz"), "rt", encoding="utf-8") as f:
        for ln in f:
            try: t = json.loads(ln)["text"]
            except Exception: continue
            a = enc(t + "\n\n"); web.append(a); nweb += len(a)
            if nweb >= WEB: break
    if nweb >= WEB: break
docs += [("web", a) for a in web]; man["web"] = dict(docs=len(web), chars=int(nweb), source="allenai/c4 en shards 1-3")
print("web", nweb, flush=True)
# code
HELD = {"scipy", "networkx"}; roots = os.path.join(sys.base_prefix, "Lib"); files = []
for dp, dn, fn in os.walk(roots):
    rel = os.path.relpath(dp, roots).replace("\\", "/"); in_sp = rel.startswith("site-packages")
    if "__pycache__" in dp: continue
    top = rel.split("/")[1] if in_sp and "/" in rel else ""
    if in_sp and top in HELD: continue
    for f in fn:
        if f.endswith(".py"):
            p = os.path.join(dp, f)
            try:
                if 200 <= os.path.getsize(p) <= 400_000: files.append(p)
            except Exception: pass
rnd.shuffle(files); seen = set(); code, ncode = [], 0
for p in files:
    if ncode >= CODE: break
    try: s = open(p, encoding="utf-8", errors="replace").read()
    except Exception: continue
    h = hashlib.md5(s.encode("utf-8", "replace")).hexdigest()
    if h in seen: continue
    seen.add(h); a = enc(s[:60_000] + "\n\n"); code.append(a); ncode += len(a)
docs += [("code", a) for a in code]; man["code"] = dict(docs=len(code), chars=int(ncode), source="Python local sin scipy/networkx, dedup md5, <=60k chars/archivo")
print("code", ncode, flush=True)
# gsm8k, dolly, shakespeare (igual que mix1)
g = []
for ln in open(os.path.join(RAW, "gsm8k", "train.jsonl"), encoding="utf-8"):
    j = json.loads(ln); g.append(enc(f"Question: {j['question']}\nAnswer: {j['answer']}\n\n"))
docs += [("math", a) for a in g]; man["math"] = dict(docs=len(g), chars=int(sum(map(len, g))))
dl = []
for ln in open(os.path.join(RAW, "dolly", "dolly.jsonl"), encoding="utf-8"):
    j = json.loads(ln); s = f"### Instruction:\n{j['instruction']}\n\n"
    if j.get("context"): s += f"### Context:\n{j['context']}\n\n"
    dl.append(s + f"### Response:\n{j['response']}\n\n")
r2 = random.Random(1234); r2.shuffle(dl); dl = dl[max(1, int(0.05 * len(dl))):]   # mismo corte que mix1 (val = 5% inicial tras barajar con semilla 1234)
dd = [enc(t) for t in dl]
docs += [("chat", a) for a in dd]; man["chat"] = dict(docs=len(dd), chars=int(sum(map(len, dd))))
sh = enc(open(os.path.join(ROOT, "data", "shakespeare", "clean.txt"), encoding="utf-8").read()); sh = sh[:int(len(sh) * 0.9)]
bk = [sh[i:i + 20_000] for i in range(0, len(sh), 20_000)]
docs += [("books", a) for a in bk]; man["books"] = dict(docs=len(bk), chars=int(len(sh)))
rnd.shuffle(docs)
total = sum(len(a) for _, a in docs); train = np.empty(total, dtype=np.uint8); o = 0
for _, a in docs: train[o:o + len(a)] = a; o += len(a)
np.save(os.path.join(OUT, "train.npy"), train)
for f in ("val.npy", "vocab.json", "val_web.npy", "val_code.npy", "val_math.npy", "val_chat.npy", "val_books.npy"): shutil.copy(os.path.join(MIX, f), os.path.join(OUT, f))
man["train_total_chars"] = int(total); man["V"] = len(itos); man["share"] = {k: round(v["chars"] / total, 4) for k, v in man.items() if isinstance(v, dict) and "chars" in v}
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1); print(json.dumps(man, indent=1))

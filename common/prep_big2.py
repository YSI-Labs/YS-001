"""Corpus 'big2' (~690M chars): corrige la CONTAMINACION de chat de big1 (se excluyen por CONTENIDO los documentos de val_chat) y reequilibra la mezcla
de dominios con sobremuestreo por repeticion de dominios pequenos (equivale a ponderar sus conteos; remedio de la dilucion medida en big1).
El tramo dev (ultimo 5% del train) se compone SOLO de documentos con una unica copia (sin repeticiones en el 95% inicial), para que los pesos de la
mezcla no se ajusten con documentos ya contados. Uso: python -I prep_big2.py"""
import os, sys, json, gzip, hashlib, random, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw_mix"); MIX = os.path.join(ROOT, "data", "mix1"); OUT = os.path.join(ROOT, "data", "big2"); os.makedirs(OUT, exist_ok=True)
REP = dict(web=1, code=1, math=6, chat=4, books=6)
WEB_CHARS, CODE_CHARS = 400_000_000, 190_000_000
rnd = random.Random(2026)
itos = json.load(open(os.path.join(MIX, "vocab.json"))); stoi = {c: i for i, c in enumerate(itos)}
lut = np.full(256, stoi["?"], dtype=np.uint8)
for c, i in stoi.items(): lut[ord(c)] = i
lut[ord("\r")] = stoi["\n"]
UNI = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-", "…": "...", " ": " ", "•": "*",
       "é": "e", "è": "e", "à": "a", "ü": "u", "ö": "o", "ä": "a", "°": " deg"}
def enc(s):
    for k, v in UNI.items(): s = s.replace(k, v)
    return lut[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
dom = {}
# web
web, n = [], 0
for i in ("00001", "00002", "00003"):
    with gzip.open(os.path.join(RAW, "c4big", f"c4_{i}.json.gz"), "rt", encoding="utf-8") as f:
        for ln in f:
            try: t = json.loads(ln)["text"]
            except Exception: continue
            a = enc(t + "\n\n"); web.append(a); n += len(a)
            if n >= WEB_CHARS: break
    if n >= WEB_CHARS: break
dom["web"] = web
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
rnd.shuffle(files); seen = set(); code, n = [], 0
for p in files:
    if n >= CODE_CHARS: break
    try: s = open(p, encoding="utf-8", errors="replace").read()
    except Exception: continue
    h = hashlib.md5(s.encode("utf-8", "replace")).hexdigest()
    if h in seen: continue
    seen.add(h); a = enc(s[:60_000] + "\n\n"); code.append(a); n += len(a)
dom["code"] = code
# math: solo train oficial
dom["math"] = [enc(f"Question: {j['question']}\nAnswer: {j['answer']}\n\n") for j in (json.loads(l) for l in open(os.path.join(RAW, "gsm8k", "train.jsonl"), encoding="utf-8"))]
# chat: EXCLUIR por contenido los documentos de val_chat (la contaminacion de big1)
valchat = open(os.path.join(MIX, "val_chat.npy"), "rb").read()      # .npy: cabecera + bytes; busqueda de subcadena de bytes
chat, excl = [], 0
for j in (json.loads(l) for l in open(os.path.join(RAW, "dolly", "dolly.jsonl"), encoding="utf-8")):
    s = f"### Instruction:\n{j['instruction']}\n\n" + (f"### Context:\n{j['context']}\n\n" if j.get("context") else "") + f"### Response:\n{j['response']}\n\n"
    a = enc(s)
    if a.tobytes() in valchat: excl += 1; continue
    chat.append(a)
dom["chat"] = chat
sh = enc(open(os.path.join(ROOT, "data", "shakespeare", "clean.txt"), encoding="utf-8").read()); sh = sh[:int(len(sh) * 0.9)]
dom["books"] = [sh[i:i + 20_000] for i in range(0, len(sh), 20_000)]
# tramo dev: 5% de documentos de cada dominio (copia unica), resto = principal con repeticion
main, dev, man = [], [], {"excluded_val_chat_docs": excl}
for d, docs in dom.items():
    rnd.shuffle(docs); k = max(1, int(0.05 * len(docs)))
    dv, mn = docs[:k], docs[k:]
    dev += dv; main += mn * REP[d]
    man[d] = dict(docs=len(docs), main_chars=int(sum(map(len, mn))), rep=REP[d], dev_chars=int(sum(map(len, dv))))
rnd.shuffle(main); rnd.shuffle(dev)
tr = np.concatenate([a for a in main] + [a for a in dev]); cut = sum(len(a) for a in main)
np.save(os.path.join(OUT, "train.npy"), tr)
for f in ("val.npy", "vocab.json", "val_web.npy", "val_code.npy", "val_math.npy", "val_chat.npy", "val_books.npy"): shutil.copy(os.path.join(MIX, f), os.path.join(OUT, f))
man["train_total_chars"] = int(len(tr)); man["main_chars_with_repetition"] = int(cut); man["dev_chars"] = int(len(tr) - cut)
man["share_main"] = {d: round(man[d]["main_chars"] * REP[d] / cut, 4) for d in dom}
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1); print(json.dumps(man, indent=1))

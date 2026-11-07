"""Prepara el corpus: quita cabecera/pie de Gutenberg, normaliza saltos de linea,
construye vocabulario de caracteres, guarda train/val como uint8.
Split contiguo: 90% train / 10% val (el final del texto, nunca visto en entrenamiento)."""
import numpy as np, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pg100.txt")
OUT = os.path.join(ROOT, "data", "shakespeare")

def main():
    os.makedirs(OUT, exist_ok=True)
    txt = open(RAW, encoding="utf-8", errors="replace").read().replace("\r\n", "\n").replace("﻿", "")
    a = txt.index("*** START OF")
    a = txt.index("\n", a) + 1
    b = txt.index("*** END OF")
    txt = txt[a:b].strip("\n") + "\n"
    # normaliza comillas tipograficas y colapsa 3+ saltos
    for k, v in {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-"}.items():
        txt = txt.replace(k, v)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.fromiter((stoi[c] for c in txt), dtype=np.uint8, count=len(txt))
    n = len(ids)
    cut = int(n * 0.9)
    np.save(os.path.join(OUT, "train.npy"), ids[:cut])
    np.save(os.path.join(OUT, "val.npy"), ids[cut:])
    json.dump(chars, open(os.path.join(OUT, "vocab.json"), "w"))
    open(os.path.join(OUT, "clean.txt"), "w", encoding="utf-8").write(txt)
    print(f"chars total={n:,}  train={cut:,}  val={n-cut:,}  V={len(chars)}")

if __name__ == "__main__":
    main()

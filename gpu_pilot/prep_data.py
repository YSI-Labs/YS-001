"""Prepara datos para el piloto: entrena un BPE a nivel de byte PROPIO (sin depender de la licencia de tokenizadores ajenos) y tokeniza a uint16.
Entradas: archivos .txt/.jsonl(.gz) (campo "text") o .parquet (columna "text"). Salidas en --out: tokenizer.json (HF), train.bin, val.bin (uint16), meta.json.
Los ultimos --val-frac de DOCUMENTOS (no de tokens) van a val: nada de val aparece en train. Separador de documentos = <|endoftext|>.
Uso: python prep_data.py --out DIR --vocab 32000 --inputs a.jsonl.gz b.parquet [--max-chars N] [--val-frac 0.005]"""
import os, sys, json, gzip, argparse, random
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def docs(paths, max_chars):
    n = 0
    for p in paths:
        if p.endswith(".parquet"):
            import pandas as pd
            for t in pd.read_parquet(p, columns=["text"])["text"]:
                yield t; n += len(t)
                if max_chars and n >= max_chars: return
        else:
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8") as f:
                if ".jsonl" in p or ".json" in p:
                    for ln in f:
                        try: t = json.loads(ln)["text"]
                        except Exception: continue
                        yield t; n += len(t)
                        if max_chars and n >= max_chars: return
                else:
                    buf = []
                    for ln in f:
                        buf.append(ln)
                        if len("".join(buf)) > 4000: t = "".join(buf); buf = []; yield t; n += len(t)
                        if max_chars and n >= max_chars: return
                    if buf: yield "".join(buf)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--vocab", type=int, default=32000); ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--max-chars", type=int, default=0); ap.add_argument("--val-frac", type=float, default=0.005); ap.add_argument("--tok-sample-chars", type=int, default=300_000_000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); all_docs = list(docs(a.inputs, a.max_chars)); random.Random(0).shuffle(all_docs)
    nval = max(1, int(len(all_docs) * a.val_frac)); val_docs, tr_docs = all_docs[:nval], all_docs[nval:]
    tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    def sample():
        n = 0
        for d in tr_docs:
            yield d; n += len(d)
            if n >= a.tok_sample_chars: return
    tk.train_from_iterator(sample(), tr); tk.save(os.path.join(a.out, "tokenizer.json")); eos = tk.token_to_id("<|endoftext|>")
    def write(name, dl):
        n = 0
        with open(os.path.join(a.out, name), "wb") as f:
            for i in range(0, len(dl), 2000):
                ids = []
                for e in tk.encode_batch(dl[i:i + 2000]): ids.extend(e.ids); ids.append(eos)
                np.asarray(ids, dtype=np.uint16).tofile(f); n += len(ids)
        return n
    ntr, nva = write("train.bin", tr_docs), write("val.bin", val_docs)
    json.dump(dict(vocab=tk.get_vocab_size(), eos=eos, train_tokens=ntr, val_tokens=nva, train_docs=len(tr_docs), val_docs=len(val_docs), chars_train=sum(map(len, tr_docs))), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("listo", ntr, "tokens de train;", nva, "de val; vocab", tk.get_vocab_size())

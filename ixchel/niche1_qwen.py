"""NICHO 1, E2: bpc de Qwen2.5-1.5B (proxy de 'calidad' por perplejidad de un LM fuerte) sobre los 200 primeros docs de niche1_docs.json.
Uso: python -I niche1_qwen.py"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import numpy as np, torch
import eval_ref_model as R
if __name__ == "__main__":
    docs = json.load(open(os.path.join(HERE, "results", "niche1_docs.json")))["docs"]
    path = os.path.join(ROOT, "data", "ref_models", [d for d in os.listdir(os.path.join(ROOT, "data", "ref_models")) if "qwen" in d.lower()][0])
    tok, model, start = R.load(path, 8); out = []; t = time.time()
    for i, d in enumerate(docs):
        b, nt = R.text_nll_bits(tok, model, start, d); out.append(b / len(d))
        if i % 20 == 0: print(i, round(out[-1], 3), f"{time.time() - t:.0f}s", flush=True)
    np.save(os.path.join(HERE, "results", "niche1_qwen_bpc.npy"), np.array(out)); print("QWEN_DONE", float(np.mean(out)), flush=True)

"""r4_loops: perplejidad de la CONTINUACION generada (y de la referencia real del mismo prompt) bajo Qwen2.5-1.5B fp32.

Para cada texto: bits por caracter de la continuacion condicionada al prompt (solo tokens de la continuacion).
Bits totales / caracteres => comparable entre tokenizadores. ppl_char = 2^bpc.
Uso: python qwen_score.py <label> [<label> ...]   (usa los 10 primeros prompts por dominio = 30 textos por variante)
Salida: results/qwen_<label>.json ; la referencia real se guarda una vez en results/qwen_ref.json (mismos prompts)."""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = __import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
QPATH = os.path.join(ROOT, "data", "ref_models", "Qwen2.5-1.5B")
LN2 = float(np.log(2.0))
NPER = 10

def load():
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(QPATH)
    model = AutoModelForCausalLM.from_pretrained(QPATH, dtype=torch.float32); model.eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return tok, model, start

@torch.no_grad()
def cont_bpc(tok, model, start, prompt, cont):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(cont, add_special_tokens=False)["input_ids"]
    ids = [start] + p + c
    lg = model(torch.tensor([ids])).logits[0, :-1].float()
    lp = torch.log_softmax(lg, -1)
    tgt = torch.tensor(ids[1:])
    nll = -lp[torch.arange(len(tgt)), tgt][len(p):]           # solo tokens de la continuacion
    return float(nll.sum() / LN2) / max(len(cont), 1), int(len(c))

def main():
    labels = sys.argv[1:]
    tok, model, start = load()
    ref_path = os.path.join(HERE, "results", "qwen_ref.json")
    ref_cache = json.load(open(ref_path)) if os.path.exists(ref_path) else {}
    for label in labels:
        g = json.load(open(os.path.join(HERE, "results", f"gen_{label}.json"), encoding="utf-8"))
        rows = [t for t in g["texts"] if t["idx"] < NPER]
        t0 = time.time(); out = []
        for t in rows:
            key = f"{t['domain']}_{t['idx']}"
            if key not in ref_cache:
                b, nt = cont_bpc(tok, model, start, t["prompt"], t["ref"])
                ref_cache[key] = dict(bpc=b, ntok=nt)
                json.dump(ref_cache, open(ref_path, "w"), indent=1)
            gb, gnt = cont_bpc(tok, model, start, t["prompt"], t["gen"])
            out.append(dict(domain=t["domain"], idx=t["idx"], gen_bpc=gb, gen_tok=gnt, ref_bpc=ref_cache[key]["bpc"]))
        res = dict(label=label, n=len(out), seconds=round(time.time() - t0, 1), texts=out)
        json.dump(res, open(os.path.join(HERE, "results", f"qwen_{label}.json"), "w"), indent=1)
        gm = np.mean([o["gen_bpc"] for o in out]); rm = np.mean([o["ref_bpc"] for o in out])
        print(label, "gen bpc", round(gm, 4), "ref bpc", round(rm, 4), "s", round(time.time() - t0), flush=True)

if __name__ == "__main__":
    main()

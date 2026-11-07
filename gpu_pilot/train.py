"""Entrenamiento del piloto: decoder tipo Llama (transformer estandar; es la via de pesos reales, NO la arquitectura Ixchel) con AdamW (control) o DoG/L-DoG (sin tasa de aprendizaje).
bf16 en CUDA (fp32 en CPU para la prueba de humo). Mismo presupuesto de tokens, mismos datos y misma semilla en ambos optimizadores. Guarda el modelo en formato HF
(LlamaForCausalLM + tokenizer) para evaluarlo SIN cambios con ixchel/eval_ref_model.py (bpc por dominio, HellaSwag/PIQA/ARC-E, HumanEval).
Uso: python train.py --data DIR --out RUNDIR --size 125m --optim adamw|dog|ldog --tokens 2.5e9 [--lr 6e-4] [--seq 1024] [--batch-tokens 524288] [--compile] [--resume]"""
import os, sys, json, math, time, argparse
import numpy as np, torch, torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # necesario con python -I
from dog import DoG
CFG = {  # mismas formas que ixchel/measure_train_cost.py (costo medido) salvo 'tiny' (prueba de humo)
    "tiny": dict(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, intermediate_size=352),
    "125m": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=12, intermediate_size=2048),
    "350m": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16, intermediate_size=2816),
    "1b": dict(hidden_size=2048, num_hidden_layers=22, num_attention_heads=32, num_key_value_heads=4, intermediate_size=5632)}
DEFAULT_LR = {"tiny": 3e-3, "125m": 6e-4, "350m": 3e-4, "1b": 2e-4}   # valores habituales de AdamW para estos tamanos [M: practica comun, no medidos aqui]


def batches(path, seq, bsz, seed):
    arr = np.memmap(path, dtype=np.uint16, mode="r"); rng = np.random.default_rng(seed); n = len(arr) - seq - 1
    while True:
        st = rng.integers(0, n, bsz); x = np.stack([arr[s:s + seq + 1] for s in st]).astype(np.int64); yield torch.from_numpy(x)


@torch.no_grad()
def evaluate(model, path, seq, dev, nb=40, bsz=16):
    model.eval(); arr = np.memmap(path, dtype=np.uint16, mode="r"); tot = 0.0; cnt = 0; rng = np.random.default_rng(123)   # mismas ventanas de val en todas las corridas
    for _ in range(nb):
        st = rng.integers(0, len(arr) - seq - 1, bsz); x = torch.from_numpy(np.stack([arr[s:s + seq + 1] for s in st]).astype(np.int64)).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = model(x[:, :-1]).logits
        tot += float(F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="sum")); cnt += x[:, 1:].numel()
    model.train(); return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--size", default="125m", choices=list(CFG)); ap.add_argument("--optim", default="adamw", choices=["adamw", "dog", "ldog"])
    ap.add_argument("--tokens", type=float, default=2.5e9); ap.add_argument("--lr", type=float, default=None); ap.add_argument("--seq", type=int, default=1024); ap.add_argument("--batch-tokens", type=int, default=524288)
    ap.add_argument("--micro-bsz", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=None); ap.add_argument("--compile", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(a.seed); meta = json.load(open(os.path.join(a.data, "meta.json")))
    cfg = LlamaConfig(vocab_size=meta["vocab"], max_position_embeddings=max(2048, a.seq), tie_word_embeddings=False, **CFG[a.size]); model = LlamaForCausalLM(cfg).to(dev)
    npar = sum(p.numel() for p in model.parameters()); accum = max(1, a.batch_tokens // (a.micro_bsz * a.seq)); step_tokens = accum * a.micro_bsz * a.seq; steps = int(a.tokens // step_tokens)
    clip = a.clip if a.clip is not None else (1.0 if a.optim == "adamw" else 0.0); lr0 = a.lr if a.lr is not None else DEFAULT_LR[a.size]
    if a.optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr0, betas=(0.9, 0.95), weight_decay=0.1, fused=dev.type == "cuda")
    else:
        opt = DoG(model.parameters(), layerwise=(a.optim == "ldog"))
    cm = torch.compile(model) if a.compile else model; step0 = 0; ck = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev); model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step0 = s["step"]
        if a.optim != "adamw":
            opt.t, opt._init, opt.rbar_g, opt.gsq_g = s["dog"]
    log = open(os.path.join(a.out, "log.jsonl"), "a"); info = dict(args=vars(a), params=npar, steps=steps, step_tokens=step_tokens, clip=clip, lr_adamw=lr0 if a.optim == "adamw" else None, device=str(dev))
    print(json.dumps(info), flush=True); log.write(json.dumps(dict(info=info)) + "\n")
    data = batches(os.path.join(a.data, "train.bin"), a.seq, a.micro_bsz, a.seed + step0); t0 = time.time(); warm = max(1, int(0.02 * steps)); model.train()
    for step in range(step0, steps):
        if a.optim == "adamw":
            lr = lr0 * (step + 1) / warm if step < warm else lr0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
            for g in opt.param_groups:
                g["lr"] = lr
        lsum = 0.0
        for _ in range(accum):
            x = next(data).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lg = cm(x[:, :-1]).logits
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1)) / accum; loss.backward(); lsum += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else 1e9)); opt.step(); opt.zero_grad(set_to_none=True)
        if not math.isfinite(lsum):
            print("PERDIDA NO FINITA; se detiene", flush=True); sys.exit(2)
        if step % 25 == 0 or step == steps - 1:
            rec = dict(step=step, tokens=(step + 1) * step_tokens, train_loss=lsum, grad_norm=gn, elapsed_s=time.time() - t0, eta_dog=getattr(opt, "last_eta", None), lr=opt.param_groups[0].get("lr"))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n")
        if (step + 1) % a.eval_every == 0 or step == steps - 1:
            v = evaluate(model, os.path.join(a.data, "val.bin"), a.seq, dev); rec = dict(step=step, tokens=(step + 1) * step_tokens, val_loss=v, val_ppl=math.exp(v))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % a.ckpt_every == 0:
            s = dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1)
            if a.optim != "adamw":
                s["dog"] = (opt.t, opt._init, opt.rbar_g, opt.gsq_g)
            torch.save(s, ck)
    hf = os.path.join(a.out, "hf"); model.save_pretrained(hf)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.data, "tokenizer.json"), eos_token="<|endoftext|>", bos_token="<|endoftext|>", pad_token="<|endoftext|>"); tok.save_pretrained(hf)
    final = evaluate(model, os.path.join(a.data, "val.bin"), a.seq, dev, nb=200)
    json.dump(dict(final_val_loss=final, params=npar, steps=steps, tokens=steps * step_tokens, train_s=time.time() - t0), open(os.path.join(a.out, "final.json"), "w"))
    print("PILOT_RUN_DONE", a.optim, round(final, 4), flush=True)


if __name__ == "__main__":
    main()

"""Mide el costo REAL (en esta CPU) de entrenar y ejecutar redes con pesos reales de distintos tamanos, para extrapolar a 1B y 7B.
Arquitectura de medicion: decoder tipo Llama con inicializacion aleatoria (el costo por token depende del numero de parametros, no del
contenido). fp32, AdamW (solo para medir el costo del paso de optimizacion; el costo de calculo no depende del optimizador elegido).
Uso: python -I measure_train_cost.py <tamano: 25m|125m|350m|1b> [tokens_por_paso=512] [pasos=3] [hilos=8]"""
import os, sys, json, time, math
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import torch
from transformers import LlamaConfig, LlamaForCausalLM
CFG = {
    "25m": dict(hidden_size=384, num_hidden_layers=8, num_attention_heads=6, num_key_value_heads=6, intermediate_size=1024),
    "125m": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, num_key_value_heads=12, intermediate_size=2048),
    "350m": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16, intermediate_size=2816),
    "1b": dict(hidden_size=2048, num_hidden_layers=22, num_attention_heads=32, num_key_value_heads=4, intermediate_size=5632),
}
if __name__ == "__main__":
    size = sys.argv[1]; toks = int(sys.argv[2]) if len(sys.argv) > 2 else 512; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    torch.set_num_threads(threads); torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32000, max_position_embeddings=2048, tie_word_embeddings=False, **CFG[size])
    t = time.time(); model = LlamaForCausalLM(cfg); t_init = time.time() - t
    n_params = sum(p.numel() for p in model.parameters()); n_nonemb = n_params - 2 * 32000 * cfg.hidden_size
    seq = 256; bs = max(1, toks // seq)
    x = torch.randint(0, 32000, (bs, seq))
    out = dict(size=size, n_params=int(n_params), n_nonembed=int(n_nonemb), tokens_per_step=int(bs * seq), threads=threads, init_s=t_init)
    # ---- inferencia (prefill) sin gradiente
    model.eval(); tt = []
    with torch.no_grad():
        for _ in range(3):
            t = time.time(); model(x); tt.append(time.time() - t)
    out["prefill_s"] = tt; out["prefill_tok_per_s"] = bs * seq / min(tt[1:])
    # ---- decodificacion token a token con KV cache (batch 1)
    with torch.no_grad():
        ids = torch.randint(0, 32000, (1, 32)); t = time.time()
        g = model.generate(ids, max_new_tokens=24, do_sample=False, min_new_tokens=24, pad_token_id=0); dt = time.time() - t
    out["decode_tok_per_s"] = 24 / dt
    # ---- entrenamiento: fwd+bwd+AdamW
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rec = []
    for s in range(steps):
        t = time.time(); loss = model(x, labels=x).loss; t1 = time.time(); loss.backward(); t2 = time.time(); opt.step(); opt.zero_grad(set_to_none=True); t3 = time.time()
        rec.append(dict(fwd=t1 - t, bwd=t2 - t1, opt=t3 - t2, total=t3 - t))
    best = min(r["total"] for r in rec[1:]) if len(rec) > 1 else rec[0]["total"]
    out["steps"] = rec; out["train_step_s"] = best; out["train_tok_per_s"] = bs * seq / best
    out["train_flops_per_s_est"] = 6 * n_nonemb * bs * seq / best   # estimacion estandar 6*N*T (sin atencion)
    out["peak_rss_gb"] = None
    try:
        import psutil; out["peak_rss_gb"] = psutil.Process().memory_info().peak_wset / 2**30
    except Exception: pass
    json.dump(out, open(os.path.join(HERE, "results", f"cost_{size}_t{threads}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("steps", "prefill_s")}), flush=True)

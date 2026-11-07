"""Compara corridas del piloto (log.jsonl) con la puerta de decision de GO_TO_MARKET.md: DoG / L-DoG frente a AdamW (la primera ruta es el control).
Uso: python compare_runs.py RUN_ADAMW RUN_DOG [RUN_LDOG ...]"""
import sys, json, os


def load(d):
    return [json.loads(ln) for ln in open(os.path.join(d, "log.jsonl")) if '"val_loss"' in ln]


runs = {os.path.basename(os.path.normpath(d)): load(d) for d in sys.argv[1:]}
ctrl = list(runs)[0]
print("| tokens | " + " | ".join(runs) + " |")
print("|---|" + "---|" * len(runs))
for i, e in enumerate(runs[ctrl]):
    if i % max(1, len(runs[ctrl]) // 12) == 0 or i == len(runs[ctrl]) - 1:
        print(f"| {e['tokens'] / 1e6:.0f}M | " + " | ".join(f"{r[i]['val_loss']:.4f}" if i < len(r) else "-" for r in runs.values()) + " |")
base = runs[ctrl][-1]["val_loss"]
for k, r in runs.items():
    if k == ctrl:
        continue
    d = r[-1]["val_loss"] - base
    print(f"{k} - {ctrl}: {d:+.4f} nats/token ->", "PASA la puerta (<= +0.05): se puede mantener la regla 'sin AdamW'" if d <= 0.05 else "NO pasa (> +0.05): usar AdamW como control para el modelo comercial")

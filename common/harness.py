"""Harness comun de evaluacion para TODAS las arquitecturas del proyecto.

Contrato de modelo (duck typing):
    class Model:
        name: str
        def fit(self, train: np.ndarray[uint8], V: int) -> None
        def predict(self, ids: np.ndarray[uint8], pos: np.ndarray[int64]) -> np.ndarray[float32, (len(pos), V)]
            # fila i = distribucion p(. | ids[:pos[i]]) NORMALIZADA (suma 1, todos > 0).
            # PROHIBIDO mirar ids[pos[i]:] (el harness lo comprueba con leak_check).
        def n_params(self) -> int      # numero de parametros/celdas de estado aprendido

Metrica principal: bits por caracter (bpc) sobre el tramo de validacion (nunca visto en fit),
evaluado en `n_eval` posiciones equiespaciadas. Sin adaptacion online durante la evaluacion.
Referencia: log2(V) ~ 6.5 bpc es el azar; unigrama ~ 4.7 bpc.
"""
import numpy as np, json, os, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", os.environ.get("IXCHEL_DATA", "shakespeare"))


def load():
    train = np.load(os.path.join(DATA, "train.npy"))
    val = np.load(os.path.join(DATA, "val.npy"))
    itos = json.load(open(os.path.join(DATA, "vocab.json")))
    return train, val, itos


def eval_positions(val, n_eval=50_000, min_ctx=64):
    n = len(val)
    return np.linspace(min_ctx, n - 1, n_eval).astype(np.int64)


def leak_check(model, val, V, n=64, seed=0):
    """Cambia el futuro (val[pos:]) por ruido y exige predicciones identicas."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(200, len(val) - 10, size=n).astype(np.int64)
    p1 = model.predict(val, pos)
    corrupted = val.copy()
    # corrompe, para cada pos, solo el futuro: usamos una copia por pos (costoso pero n pequeno)
    ok = True
    for i, p in enumerate(pos):
        c = val.copy()
        c[p:] = rng.integers(0, V, size=len(c) - p).astype(np.uint8)
        pi = model.predict(c, np.array([p], dtype=np.int64))
        if not np.allclose(pi[0], p1[i], atol=1e-5):
            ok = False
            break
    return ok


def evaluate(model, val, V, n_eval=50_000, batch=4096, check_leak=True):
    pos = eval_positions(val, n_eval)
    t0 = time.time()
    nll = 0.0
    worst_norm = 0.0
    minp = 1.0
    for s in range(0, len(pos), batch):
        pp = pos[s:s + batch]
        P = model.predict(val, pp).astype(np.float64)
        assert P.shape == (len(pp), V), f"forma de predict incorrecta: {P.shape}"
        worst_norm = max(worst_norm, float(np.abs(P.sum(1) - 1).max()))
        minp = min(minp, float(P.min()))
        p_true = P[np.arange(len(pp)), val[pp]]
        nll += float(-np.log2(np.maximum(p_true, 1e-12)).sum())
    bpc = nll / len(pos)
    res = dict(bpc=bpc, max_norm_err=worst_norm, min_prob=minp,
               eval_time_s=time.time() - t0, eval_chars_per_s=len(pos) / (time.time() - t0))
    if check_leak:
        res["leak_free"] = bool(leak_check(model, val, V))
    return res


def generate(model, prompt_ids, n, V, seed=0, itos=None, max_ctx=None):
    """Muestreo ancestral puro (temperatura 1.0 = sin hiperparametros de muestreo)."""
    rng = np.random.default_rng(seed)
    buf = np.zeros(len(prompt_ids) + n, dtype=np.uint8)
    buf[:len(prompt_ids)] = prompt_ids
    L = len(prompt_ids)
    for i in range(n):
        p = model.predict(buf, np.array([L], dtype=np.int64))[0].astype(np.float64)
        p = p / p.sum()
        buf[L] = rng.choice(V, p=p)
        L += 1
    out = buf[:L]
    return "".join(itos[i] for i in out) if itos else out


def run(model, n_eval=50_000, gen_chars=600, seed=0, extra=None, out_json=None):
    """fit + evaluate + generate + reporte. Devuelve dict con todo."""
    train, val, itos = load()
    V = len(itos)
    c0, w0 = time.process_time(), time.time()
    model.fit(train, V)
    fit_cpu, fit_wall = time.process_time() - c0, time.time() - w0
    res = evaluate(model, val, V, n_eval=n_eval)
    prompt = val[1000:1064]
    sample = generate(model, prompt, gen_chars, V, seed=seed, itos=itos)
    res.update(name=getattr(model, "name", type(model).__name__), fit_cpu_s=fit_cpu, fit_wall_s=fit_wall,
               n_params=int(model.n_params()), train_chars=len(train),
               train_chars_per_s=len(train) / max(fit_wall, 1e-9),
               prompt="".join(itos[i] for i in prompt), sample=sample)
    if extra:
        res.update(extra)
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        json.dump(res, open(out_json, "w"), indent=1)
    return res


def summarize(res):
    print(f"[{res['name']}] bpc={res['bpc']:.4f}  params={res['n_params']:,}  "
          f"fit={res['fit_wall_s']:.1f}s (cpu {res['fit_cpu_s']:.1f}s, {res['train_chars_per_s']:,.0f} ch/s)  "
          f"leak_free={res.get('leak_free')}  norm_err={res['max_norm_err']:.1e}")
    print("--- muestra ---")
    print(res["prompt"] + "|" + res["sample"][len(res["prompt"]):][:600])

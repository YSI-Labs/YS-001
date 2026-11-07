"""R4 gate: mezcla PoE con compuerta por dominio sobre mix15 (IXCHEL_DATA=mix15).
Uso: python -I run_gate.py <tag> [--mc]
Pasos (un solo ajuste de expertos; todas las mezclas comparten las mismas salidas de expertos):
  1. Tablas de cadenas KN sobre train[:cut]  -> expertos sobre el dev interno (5% final del train) -> LPdev.
  2. Compuerta causal de dominio (k-means sobre rasgos de 500 chars, k por BIC) + compuerta 'nivel A' (ya en core).
  3. Pesos PoE por variante: global | cl | alev | cl_alev, con encogimiento EB (gatepoe.fit_gated).
  4. Seleccion de variante SOLO con dev: CV en 2 pliegues por bloques de 500 posiciones del dev.
  5. Tablas sobre train completo y evaluacion en val_* (30k posiciones, mismas que mix15_base) con NLL pareado.
  6. Opcional --mc: opcion multiple (HellaSwag n=1500, PIQA n=800, ARC-E n=1000) con la variante elegida.
"""
import os, sys, json, time
os.environ["IXCHEL_DATA"] = "mix15"; os.environ.setdefault("NUMBA_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "common"))
import numpy as np, harness, eval_suite as ES
from core_v04 import Ixchel, default_chains, unigram
from longcache import KeyedWin
import gatepoe as GP

INF = 1 << 40
LEX = dict(Kb=[[], ["c1"], ["c1", "c3"], ["c1", "c2", "c3", "c4"]],
           Kc=[[], ["c1", "c2"], ["c1", "c2", "c4", "c5"], ["c1", "c2", "c3", "c4", "c5", "c6"]],
           WL=[[], ["wlen"], ["wlen", "c1", "c2"]], W2=[[], ["wp"], ["wp", "pw2"]])
VARIANTS = ("global", "cl", "alev", "cl_alev")
DOMAINS = ("web", "code", "math", "chat", "books")
BLOCK = 500
LN2 = np.log(2.0)


def base_model():  # misma configuracion que ixchel/run_mix15.py base_model()
    ch = {**default_chains(), **LEX}
    return Ixchel(name="mix15", verbose=False, chains=ch, mix="poe", cache_windows=(),
                  xperts=(KeyedWin("C8", [("c", k) for k in range(1, 9)], windows=[256, 1024, 4096, INF]),))


def gate_arr(variant, cl, alev):
    if variant == "global": return np.zeros(len(alev), np.int64)
    if variant == "cl": return cl
    if variant == "alev": return alev
    return cl * 10 + alev


def gate_card(variant, k):
    return {"global": 1, "cl": k, "alev": 10, "cl_alev": 10 * k}[variant]


def experts(m, ids, pos, V):
    """LP [K,m,V] float32 y nivel A (alev) de cada posicion, con historial = ids[:pos] (causal)."""
    A = m._atoms_for(ids)
    ctx = m._ctx_prepare(ids, int(pos.max()), m.base, None)
    for e in m.xperts: e.prepare(ids)
    Ps, fe = m._expert_dists(m.tabs, A, pos, ids, pos, V, ctx)
    LP = np.log(np.maximum(np.stack(Ps), 1e-12)).astype(np.float32)
    return LP, (fe[0][0] // 4).astype(np.int64)


class GatedPredictor:
    """Contrato harness.predict para la variante elegida (para opcion multiple)."""
    def __init__(self, m, gate, V, variant, W):
        self.m, self.gate, self.V, self.variant, self.W = m, gate, V, variant, W
    def predict(self, ids, pos):
        pos = np.asarray(pos, dtype=np.int64)
        LP, alev = experts(self.m, ids, pos, self.V)
        cl = self.gate.label(ids, pos) if self.variant in ("cl", "cl_alev") else np.zeros(len(pos), np.int64)
        g = gate_arr(self.variant, cl, alev)
        return np.exp(GP.mix_logp(LP, g, self.W)).astype(np.float32)


def fold_cv(LPd, xd, gate_d, G, block_id):
    """Dos pliegues por bloques alternos de dev: ajusta en un pliegue, mide NLL (bits) en el otro."""
    tot = 0.0; n = 0
    for f in (0, 1):
        tr = np.flatnonzero(block_id % 2 != f); te = np.flatnonzero(block_id % 2 == f)
        W, _ = GP.fit_gated(LPd[:, tr, :], xd[tr], gate_d[tr], G)
        L = GP.mix_logp(LPd[:, te, :], gate_d[te], W)
        tot += float(-L[np.arange(len(te)), xd[te]].sum()); n += len(te)
    return tot / n / LN2


def main(tag, do_mc):
    t0 = time.time()
    out_dir = os.path.join(HERE, "results")
    train, val_dummy, itos = harness.load(); V = len(itos)
    enc = ES.make_encoder(itos)
    n = len(train); m = base_model(); cut = int(n * (1 - m.dev_frac))
    nl, sp = itos.index("\n"), itos.index(" ")
    print(f"[{tag}] carga {time.time()-t0:.0f}s n={n} cut={cut}", flush=True)

    # ---- 1. expertos sobre el dev interno (tablas de train[:cut]), como core_v04.fit (rama no prebuilt)
    A_tr = m._atoms_for(train)
    tabs_dev = m._count(A_tr, train, m.t0, cut)
    print(f"[{tag}] tablas dev {time.time()-t0:.0f}s", flush=True)
    dev_pos = np.linspace(cut, n - 1, min(m.dev_positions, n - cut)).astype(np.int64)
    ids_dev = train[cut:]; q = dev_pos - cut
    base_dev = unigram(train[:cut], V)
    ctx = m._ctx_prepare(ids_dev, int(q.max()), base_dev, None)
    for e in m.xperts:
        if e.V is None: e.set_vocab(V, m.isletter)
        e.prepare(ids_dev)
    K = len(m.expert_names())
    LPd = np.zeros((K, len(dev_pos), V), np.float32); alev_d = np.zeros(len(dev_pos), np.int64)
    for c0 in range(0, len(dev_pos), m.chunk):
        sl = slice(c0, min(c0 + m.chunk, len(dev_pos)))
        Ps, fe = m._expert_dists(tabs_dev, A_tr, dev_pos[sl], ids_dev, q[sl], V, ctx)
        for k, p in enumerate(Ps):
            LPd[k, sl] = np.log(np.maximum(p, 1e-12))
        alev_d[sl] = fe[0][0] // 4
    xd = np.asarray(train[dev_pos]).astype(np.int64)
    names = m.expert_names()
    print(f"[{tag}] expertos dev {time.time()-t0:.0f}s  K={K}", flush=True)
    del tabs_dev, ctx

    # ---- 2. compuerta causal de dominio (solo train[:cut]; no usa dev ni val)
    gate = GP.CausalDomainGate(m.cls, nl, sp, W=500, kmax=8).fit(train, cut)
    print(f"[{tag}] compuerta: k={gate.k} BIC={gate.bics}", flush=True)
    cl_d = gate.label(ids_dev, q)
    print(f"[{tag}] reparto dev cl: {np.bincount(cl_d, minlength=gate.k).tolist()}  alev: {np.bincount(alev_d, minlength=10).tolist()}", flush=True)
    block_id = np.arange(len(dev_pos)) // BLOCK

    # ---- 3-4. pesos por variante y seleccion por CV en dev
    G_of = {v: gate_card(v, gate.k) for v in VARIANTS}
    Wv, info, cv = {}, {}, {}
    for v in VARIANTS:
        gd = gate_arr(v, cl_d, alev_d)
        Wv[v], info[v] = GP.fit_gated(LPd, xd, gd, G_of[v], verbose=True)
        cv[v] = fold_cv(LPd, xd, gd, G_of[v], block_id)
        Lin = GP.mix_logp(LPd, gd, Wv[v])
        info[v]["dev_bpc_insample"] = float(-Lin[np.arange(len(xd)), xd].mean() / LN2)
        info[v]["dev_bpc_cv"] = cv[v]
        print(f"[{tag}] {v}: G={G_of[v]} dev in-sample={info[v]['dev_bpc_insample']:.4f} dev-CV={cv[v]:.4f} "
              f"tau2={info[v]['tau2']:.5f} t={time.time()-t0:.0f}s", flush=True)
    chosen = min(VARIANTS, key=lambda v: cv[v])
    print(f"[{tag}] variante elegida por dev-CV: {chosen}", flush=True)
    del LPd

    # ---- 5. tablas sobre train completo + evaluacion val (no se usa para elegir)
    m.base = unigram(train, V)
    m.tabs = m._count(A_tr, train, m.t0, n)
    print(f"[{tag}] tablas finales {time.time()-t0:.0f}s", flush=True)
    base_nll = {}
    bz = os.path.join(ROOT, "ixchel", "results", "mix15_base_nll.npz")
    if os.path.exists(bz):
        base_nll = dict(np.load(bz))
    res = dict(tag=tag, chain_gate=dict(k=gate.k, bics=gate.bics), chosen_by_dev_cv=chosen, variants={})
    nll_store = {}
    for d in DOMAINS:
        val = np.load(os.path.join(ROOT, "data", "mix15", f"val_{d}.npy"))
        pos = harness.eval_positions(val, min(30_000, len(val) - 100))
        LP, alev = experts(m, val, pos, V)
        cl = gate.label(val, pos)
        xs = val[pos].astype(np.int64)
        for v in VARIANTS:
            L = GP.mix_logp(LP, gate_arr(v, cl, alev), Wv[v])
            nll = -L[np.arange(len(xs)), xs] / LN2
            nll_store.setdefault(v, {})[d] = nll
        del LP
        print(f"[{tag}] dominio {d} listo {time.time()-t0:.0f}s", flush=True)

    for v in VARIANTS:
        vr = dict(G=G_of[v], W=Wv[v].tolist(), **{k: info[v][k] for k in ("tau2", "lam", "n", "dev_bpc_insample", "dev_bpc_cv")})
        vr["domains"] = {}
        for d in DOMAINS:
            nll = nll_store[v][d]
            nb = len(nll) // BLOCK
            rec = dict(bpc=float(nll.mean()), n=len(nll))
            if d in base_nll and len(base_nll[d]) == len(nll):
                diff = nll - base_nll[d]
                bm = diff[:nb * BLOCK].reshape(nb, BLOCK).mean(1)
                rec.update(delta_vs_base=float(diff.mean()), se_block=float(bm.std(ddof=1) / np.sqrt(nb)),
                           base_bpc=float(base_nll[d].mean()))
                rec["z"] = rec["delta_vs_base"] / rec["se_block"] if rec["se_block"] > 0 else 0.0
            vr["domains"][d] = rec
        res["variants"][v] = vr
        np.savez(os.path.join(out_dir, f"gate_{tag}_{v}_nll.npz"), **nll_store[v])
    json.dump(res, open(os.path.join(out_dir, f"gate_{tag}.json"), "w"), indent=1)
    print(json.dumps({v: {d: round(res["variants"][v]["domains"][d]["bpc"], 4) for d in DOMAINS} for v in VARIANTS}), flush=True)

    # ---- 6. opcion multiple con la variante elegida por dev
    if do_mc:
        gp = GatedPredictor(m, gate, V, chosen, Wv[chosen])
        mc = {}
        mc["hellaswag"] = ES.hellaswag(gp, enc, V, n=1500); print(f"[{tag}] hellaswag acc_norm {mc['hellaswag']['acc_norm']}", flush=True)
        mc["piqa"] = ES.piqa(gp, enc, V, n=800); print(f"[{tag}] piqa acc_norm {mc['piqa']['acc_norm']}", flush=True)
        mc["arc_easy"] = ES.arc_easy(gp, enc, V, n=1000); print(f"[{tag}] arc acc_norm {mc['arc_easy']['acc_norm']}", flush=True)
        res["mc_chosen"] = mc
        json.dump(res, open(os.path.join(out_dir, f"gate_{tag}.json"), "w"), indent=1)
    print(f"[{tag}] fin {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], "--mc" in sys.argv)

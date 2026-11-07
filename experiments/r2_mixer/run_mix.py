"""Corrida de ablaciones de la mezcla. 1) ajusta todas las cabezas en dA/dB (tablas en train[:cut]);
2) elige la cabeza por dev-C (tramo held-out, nunca val); 3) reporta val (n_eval=20000) de todas las cabezas
con harness.evaluate (sin leak check, para ahorrar tiempo); 4) corre harness.run completo (leak check, generacion)
sobre la cabeza elegida, sin reajustar."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import mixer
import harness

if __name__ == "__main__":
    n_eval = 20_000
    train, val, itos = harness.load()
    V = len(itos)
    m = mixer.Mix(name="Ixchel-r2-mixer")
    t0 = time.time()
    m.fit(train, V)
    fit_s = time.time() - t0
    out = {"fit_s": fit_s, "dev_C_bpc": m.dev_report, "heads": {}}
    names = list(m.heads.keys())
    for nm in names:
        m.head = m.heads[nm]
        r = harness.evaluate(m, val, V, n_eval=n_eval, check_leak=False)
        out["heads"][nm] = dict(val_bpc=r["bpc"], eval_chars_per_s=r["eval_chars_per_s"], n_params=int(m.n_params()),
                                dev_C_bpc=m.dev_report[nm])
        print(f"{nm:20s} dev-C {m.dev_report[nm]:.4f}  val20k {r['bpc']:.4f}  ch/s {r['eval_chars_per_s']:,.0f}", flush=True)
    best = min(names, key=lambda k: m.dev_report[k])
    out["chosen_by_dev_C"] = best
    out["eb_info"] = dict(hier_tau2=m.heads["hier_fine"].get("tau2"), hier_mean_strength=m.heads["hier_fine"].get("eb_mean"),
                          poe_global_info=m.heads["poe_global"].get("poe"))
    json.dump(out, open(os.path.join(HERE, "mixer_ablation.json"), "w"), indent=1)
    print("elegida por dev-C:", best, flush=True)
    m.head = m.heads[best]
    res = harness.run(mixer.Fixed(m, name=f"Ixchel-r2-mixer[{best}]"), n_eval=n_eval, gen_chars=400,
                      out_json=os.path.join(HERE, f"final_{best}.json"))
    res.update(fit_time_s=m.fit_time, chosen_head=best, dev_C_bpc=m.dev_report)
    json.dump(res, open(os.path.join(HERE, f"final_{best}.json"), "w"), indent=1)
    harness.summarize(res)

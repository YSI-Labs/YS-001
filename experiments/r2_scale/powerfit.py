import json, numpy as np
from scipy.optimize import curve_fit
R = "results/"
out = {}
for var in ["A", "AWSL"]:
    fr = [0.06, 0.125, 0.25, 0.5, 1.0]
    d = [json.load(open(f"{R}scale_{var}_{f}.json")) for f in fr]
    N = np.array([x["train_chars"] for x in d], float)
    y = np.array([x["bpc"] for x in d])
    P = np.array([x["n_params_pairs"] for x in d], float)
    f = lambda N, a, b, c: a * N ** (-b) + c
    try:
        popt, pcov = curve_fit(f, N, y, p0=[5, 0.2, 2.0], maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        res = y - f(N, *popt)
        out[var] = dict(a=popt[0], b=popt[1], c=popt[2], se=perr.tolist(), resid=res.tolist())
        for k in (10, 100, 1000):
            out[var][f"x{k}"] = float(f(N[-1] * k, *popt))
        out[var]["floor_c"] = popt[2]
    except Exception as e:
        out[var] = dict(error=str(e))
    # ley de potencia simple sin piso: bpc = a N^-b
    lp = np.polyfit(np.log(N), np.log(y - 1.0), 1) if False else None
    # pendiente log-log de pares vs N
    out[var]["pairs_per_char"] = float(np.polyfit(np.log(N), np.log(P), 1)[0])
    out[var]["pairs_exp_vs_chars"] = float(np.polyfit(np.log(N), np.log(P), 1)[0])
    # ajuste de 2 parametros (sin piso) en los 3 puntos mas grandes para comparar
    try:
        g = lambda N, a, b: a * N ** (-b)
        p2, _ = curve_fit(g, N[2:], y[2:], p0=[5, 0.1])
        out[var]["noFloor_last3"] = dict(a=p2[0], b=p2[1])
    except Exception as e:
        out[var]["noFloor_last3"] = str(e)
print(json.dumps(out, indent=1))
json.dump(out, open(R + "powerfit.json", "w"), indent=1)

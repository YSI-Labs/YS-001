"""Baseline de calibracion: n-gram de caracteres con interpolacion Witten-Bell (sin hiperparametros
salvo el orden maximo). Entrenamiento = contar (ordenar claves). Sirve de vara de medir."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness


class WittenBell:
    def __init__(self, order=6):
        self.N = order
        self.name = f"WB-ngram-{order}"

    def fit(self, train, V):
        self.V = V
        x = train.astype(np.int64)
        n = len(x)
        self.uni = np.bincount(x, minlength=V).astype(np.float64)
        self.tabs = []  # por orden k>=1: (uctx, tot, typ, ukeys, ucnt)
        ctx = np.zeros(n, dtype=np.int64)  # ctx[i] = clave del contexto de longitud k que termina en i-1
        for k in range(1, self.N + 1):
            # ctx_k[i] = ctx_{k-1}[i]*? ... construimos por desplazamiento: clave = sum x[i-k+j] V^(k-1-j)
            # ctx_k[i] = x[i-k]*V^(k-1) + ctx_{k-1}[i]  (con ctx_{k-1}[i] sobre las k-1 mas recientes)
            ck = np.zeros(n, dtype=np.int64)
            ck[k:] = x[:n - k] * (self.V ** (k - 1)) + ctx[k:]
            ctx = ck
            keys = ctx[k:] * V + x[k:]
            ukeys, ucnt = np.unique(keys, return_counts=True)
            uc = ukeys // V
            uctx, idx, typ = np.unique(uc, return_index=True, return_counts=True)
            tot = np.add.reduceat(ucnt, idx)
            self.tabs.append((uctx, tot.astype(np.float64), typ.astype(np.float64), ukeys, ucnt.astype(np.float64)))

    def n_params(self):
        return int(sum(len(t[3]) for t in self.tabs))

    def predict(self, ids, pos, order=None):
        V = self.V
        N = self.N if order is None else order
        pos = np.asarray(pos, dtype=np.int64)
        m = len(pos)
        U0 = float((self.uni > 0).sum())
        p = (self.uni[None, :] + U0 / V) / (self.uni.sum() + U0)
        p = np.repeat(p, m, axis=0)
        x = ids.astype(np.int64)
        key = np.zeros(m, dtype=np.int64)
        for k in range(1, N + 1):
            key = key + x[pos - k] * (V ** (k - 1))
            uctx, tot, typ, ukeys, ucnt = self.tabs[k - 1]
            j = np.searchsorted(uctx, key)
            j = np.minimum(j, len(uctx) - 1)
            found = uctx[j] == key
            if not found.any():
                continue
            fi = np.nonzero(found)[0]
            kk = key[fi][:, None] * V + np.arange(V)[None, :]
            jj = np.searchsorted(ukeys, kk.ravel())
            jj = np.minimum(jj, len(ukeys) - 1)
            hit = (ukeys[jj] == kk.ravel())
            cnt = np.where(hit, ucnt[jj], 0.0).reshape(len(fi), V)
            T = tot[j[fi]][:, None]
            U = typ[j[fi]][:, None]
            p[fi] = (cnt + U * p[fi]) / (T + U)
        return (p / p.sum(1, keepdims=True)).astype(np.float32)


if __name__ == "__main__":
    train, val, itos = harness.load()
    V = len(itos)
    best = None
    for order in (3, 5, 7, 8):
        m = WittenBell(order)
        res = harness.run(m, n_eval=20_000, gen_chars=400, out_json=os.path.join(harness.ROOT, "experiments", "baselines", f"wb{order}.json"))
        harness.summarize(res)

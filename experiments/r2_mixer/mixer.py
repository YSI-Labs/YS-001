"""r2_mixer: mejoras de la MEZCLA de Ixchel (sin learning rate libre).

Reutiliza de core.py (copia de ixchel/core.py) los expertos: tablas KN, distribuciones de cada cadena
(_count, _chain_dists) y atomos. Aqui se reemplaza SOLO la combinacion:

  H0 em_base       : mezcla lineal + compuerta base (A lhi/nb, W lhi/nb) por EM (= v0.1)
  H1 em_global     : mezcla lineal sin compuerta (ablacion)
  H2 em_flat_fine  : mezcla lineal con compuerta FINA plana (base x S x L), sin jerarquia (ablacion de dispersion)
  H3 poe_global    : producto de expertos p ~ prod_k p_k^{w_k} sin compuerta (ablacion)
  H4 poe_base      : producto de expertos con pesos por compuerta base; ajuste por Newton + busqueda de linea
                     Armijo; prior gaussiano EB hacia los pesos globales (precision estimada de datos)
  H5 hier_fine     : compuerta fina (base x S-lhi/nb x L-lhi/nb) con EM y shrinkage EB hacia el padre base
  H6 stack2        : 2 capas: 5 mezclas de capa 1 (compuertas base, c1, clase(k1)xlhW, S, L) + EM de capa 2
                     sobre sus salidas (capa 1 ajustada en tramo dA, capa 2 en tramo dB)
  +sse             : etapa SSE/APM por cuantiles de p (9 contextos = clase de char previo), ajustada en dB

Particion del tramo dev interno (ultimo 5% de train, tablas contadas en train[:cut]):
  dA = 1/3 (ajuste de pesos), dB = 1/3 (capa 2 y SSE), dC = 1/3 (seleccion por dev, nunca val).
Val solo se usa via harness.evaluate / harness.run.
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")), 'common'))
import harness
from core import Ixchel, default_chains, _nbucket, ATOM_NAMES


class Mix(Ixchel):
    def __init__(self, name="Ixchel-mix", dev_positions=150_000, em_iters=60, prior_m=1.0, sse_bins=24, verbose=True):
        super().__init__(chains=default_chains(), gate_chains=("A", "W"), name=name, em_iters=em_iters,
                         prior_m=prior_m, dev_frac=0.05, dev_positions=dev_positions, gated=True,
                         verbose=verbose, t0=16)
        self.iA, self.iW, self.iS, self.iL = (self.names.index(c) for c in ("A", "W", "S", "L"))
        self.c1col = int(np.flatnonzero(self.cols == ATOM_NAMES["c1"])[0])
        self.k1col = int(np.flatnonzero(self.cols == ATOM_NAMES["k1"])[0])
        self.radix = {nm: (len(self.chains[nm]) + 1) * 4 for nm in self.names}
        self.sse_bins = sse_bins
        self.head = None
        self.do_poe_base = False
        self.heads = {}
        self.dev_report = {}

    # ------------------------------------------------------------ compuertas
    def gates(self, Ar, LH, NN):
        """Ar [m, ncols] atomos de las posiciones; LH/NN listas por cadena. Devuelve dict de ids de compuerta."""
        RA, RW, RS, RL = self.radix["A"], self.radix["W"], self.radix["S"], self.radix["L"]
        f = {ci: (LH[ci] + 1) * 4 + _nbucket(NN[ci]) for ci in (self.iA, self.iW, self.iS, self.iL)}
        base = f[self.iA] * RW + f[self.iW]                       # G = RA*RW = 800
        fine = base * (RS * RL) + f[self.iS] * RL + f[self.iL]    # G = 800*RS*RL
        c1 = Ar[:, self.c1col].astype(np.int64)                   # 0..255 (255 = sin caracter)
        cls = Ar[:, self.k1col].astype(np.int64) * 5 + (LH[self.iW] + 1)   # clase k1 (0..8) x lhi_W (0..4)
        return dict(base=base.astype(np.int64), fine=fine.astype(np.int64), c1=c1, cls=cls,
                    S=f[self.iS].astype(np.int64), L=f[self.iL].astype(np.int64))

    GSIZE = dict(base=800, fine=800 * 24 * 20, c1=256, cls=45, S=24, L=20)

    # ------------------------------------------------------------ estimadores
    def _em_w(self, pt, g, G):
        return self._em(pt, g, G)                                  # EM v0.1 (prior_m pseudo-cuentas al global)

    def em_hier(self, pt, g, G, Wp, Rr, iters):
        """EM de la compuerta fina; cada hijo se encoge hacia el padre Wp[g // Rr] con fuerza EB:
        tau2_k = (sum_c n_c d_ck^2 - sum_c v_ck) / sum_c n_c  (v = wp(1-wp)), m_k = wp(1-wp)/tau2_k - 1 (Beta)."""
        K = pt.shape[1]
        cpar = np.arange(G) // Rr
        cnt = np.bincount(g, minlength=G).astype(np.float64)
        wpc = Wp[cpar]
        vb = wpc * (1 - wpc)
        W = wpc.copy()
        has = cnt > 0
        tau2 = np.full(K, 1e-8)
        mk = np.zeros_like(wpc)
        for it in range(iters):
            num = W[g] * pt
            r = num / np.maximum(num.sum(1, keepdims=True), 1e-300)
            acc = np.stack([np.bincount(g, weights=r[:, k], minlength=G) for k in range(K)], axis=1)
            n = cnt[:, None]
            d = acc / np.maximum(n, 1e-12) - wpc
            tau2 = np.maximum(((n * d * d)[has].sum(0) - vb[has].sum(0)) / max(cnt[has].sum(), 1e-12), 1e-8)
            mk = np.maximum(vb / tau2[None, :] - 1.0, 0.0)
            den = n + mk
            W = np.where(den > 0, (acc + mk * wpc) / np.maximum(den, 1e-300), wpc)
        self.last_tau2 = tau2.tolist()
        self.last_eb_strength_mean = float(mk[has].mean()) if has.any() else 0.0
        return W

    def poe_newton(self, LP, xs, g, G, w_init, w0, lam, iters=25):
        """Maximiza por gate F_g(w) = sum_i [w.s_i - log Z_i(w)] - (1/2) sum_k lam_k (w_k - w0_k)^2 .
        s_ik = log p_k(x_i), Z_i(w) = sum_x prod_k p_k(x)^{w_k}. F es concava en w (log-sum-exp convexa).
        Newton con matriz de Hessiana exacta (cov. bajo q_i) y busqueda de linea Armijo (c=1e-4, reduccion 1/2)."""
        K, m, V = LP.shape
        ar = np.arange(m)
        s = np.stack([LP[k, ar, xs] for k in range(K)], axis=1)
        W = np.tile(np.asarray(w_init, np.float64), (G, 1))

        def row_obj(Wr):
            L = Wr[:, 0:1] * LP[0]
            for k in range(1, K):
                L += Wr[:, k:k + 1] * LP[k]
            mx = L.max(1)
            lse = mx + np.log(np.exp(L - mx[:, None]).sum(1))
            return (Wr * s).sum(1) - lse, L, lse

        def gate_obj(Wg):
            ro, _, _ = row_obj(Wg[g])
            return np.bincount(g, weights=ro, minlength=G) - 0.5 * (lam[None, :] * (Wg - w0[None, :]) ** 2).sum(1)

        H = None
        for it in range(iters):
            Wr = W[g]
            _, L, lse = row_obj(Wr)
            q = np.exp(L - lse[:, None])
            E = np.stack([(q * LP[k]).sum(1) for k in range(K)], axis=1)
            C = np.zeros((m, K, K))
            for j in range(K):
                qj = q * LP[j]
                for k in range(j, K):
                    C[:, j, k] = (qj * LP[k]).sum(1) - E[:, j] * E[:, k]
                    C[:, k, j] = C[:, j, k]
            R = s - E
            grad = np.stack([np.bincount(g, weights=R[:, k], minlength=G) for k in range(K)], axis=1) \
                - lam[None, :] * (W - w0[None, :])
            H = np.zeros((G, K, K))
            for j in range(K):
                for k in range(K):
                    H[:, j, k] = np.bincount(g, weights=C[:, j, k], minlength=G)
            H += np.diag(lam)[None] + 1e-6 * np.eye(K)[None]
            D = np.einsum("gjk,gk->gj", np.linalg.pinv(H), grad)
            f0 = gate_obj(W)
            slope = (grad * D).sum(1)
            alpha = np.ones(G)
            done = np.zeros(G, bool)
            for _ in range(8):
                cand = W + alpha[:, None] * D
                f1 = gate_obj(cand)
                ok = (~done) & (f1 >= f0 + 1e-4 * alpha * slope)
                W[ok] = cand[ok]
                done |= ok
                if done.all():
                    break
                alpha = np.where(done, alpha, alpha * 0.5)
            if np.abs(D).max() < 1e-7:
                break
        return W, H

    def poe_fit(self, LP, xs, g, G):
        K = LP.shape[0]
        zero = np.zeros(K)
        ones = np.full(K, 1.0 / K)
        W0, _ = self.poe_newton(LP, xs, np.zeros(LP.shape[1], np.int64), 1, ones, ones, zero)
        w0 = W0[0]
        # precision EB: varianza muestral por gate (Hessiana inversa) y dispersion entre gates
        Wu, Hu = self.poe_newton(LP, xs, g, G, w0, w0, zero)
        cnt = np.bincount(g, minlength=G).astype(np.float64)
        has = cnt > 0
        Vd = np.stack([np.diag(np.linalg.pinv(Hu[i])) for i in np.flatnonzero(has)])     # [#gates, K]
        dev2 = (Wu[has] - w0[None, :]) ** 2
        tau2 = np.maximum(((cnt[has][:, None] * (dev2 - Vd)).sum(0)) / cnt[has].sum(), 1e-8)
        lam = 1.0 / tau2
        Ws, _ = self.poe_newton(LP, xs, g, G, w0, w0, lam)
        self.last_poe = dict(w_global=w0.tolist(), tau2=tau2.tolist(), lam=lam.tolist())
        return Ws

    # ------------------------------------------------------------ salida de una cabeza
    def head_forward(self, head, P, G):
        """P: lista K de [m,V] (probabilidades); G: dict de compuertas. Devuelve [m,V] normalizado."""
        kind = head["kind"]
        if kind == "sse":
            base = self.head_forward(head["base"], P, G)
            return self.sse_apply(base, G["cls"] // 5, head["cal"])
        if kind == "em":
            Wm = head["W"][G[head["g"]]]
            out = np.zeros_like(P[0], dtype=np.float64)
            for k in range(len(P)):
                out += Wm[:, k:k + 1] * P[k]
            return out
        if kind == "poe":
            Wm = head["W"][G[head["g"]]]
            L = np.zeros_like(P[0], dtype=np.float64)
            for k in range(len(P)):
                L += Wm[:, k:k + 1] * np.log(np.maximum(P[k], 1e-12))
            L -= L.max(1, keepdims=True)
            e = np.exp(L)
            return e / e.sum(1, keepdims=True)
        if kind == "stack":
            qs = []
            for (gname, Wj) in head["layer1"]:
                Wm = Wj[G[gname]]
                q = np.zeros_like(P[0], dtype=np.float64)
                for k in range(len(P)):
                    q += Wm[:, k:k + 1] * P[k]
                qs.append(q)
            W2 = head["W2"][G[head["g2"]]]
            out = np.zeros_like(qs[0])
            for j, q in enumerate(qs):
                out += W2[:, j:j + 1] * q
            return out
        raise ValueError(kind)

    def sse_fit(self, out, xs, ctx):
        """Calibracion por cuantiles de log p (B bins, cuantiles de datos) x contexto (9 clases).
        Cada evento (fila, simbolo) con score q=p(x) y etiqueta hit=[x==x_i]. Tasa por bin con encogimiento
        hacia la tasa global; pseudo-cuenta = ocupacion media de las celdas (derivada de datos)."""
        m, V = out.shape
        lq = np.log(np.maximum(out, 1e-12))
        hit = np.zeros((m, V))
        hit[np.arange(m), xs] = 1.0
        lqf, hf = lq.ravel(), hit.ravel()
        cf = np.repeat(ctx, V)
        B = self.sse_bins
        edges = np.quantile(lqf, np.linspace(0, 1, B + 1)[1:-1])
        b = np.searchsorted(edges, lqf)
        nb = np.bincount(b, minlength=B).astype(np.float64)
        hb = np.bincount(b, weights=hf, minlength=B)
        keep = nb > 0
        cb = np.bincount(b, weights=lqf, minlength=B)[keep] / nb[keep]
        rate = hb / np.maximum(nb, 1)
        NC = 9
        ncell = np.bincount(cf.astype(np.int64) * B + b, minlength=NC * B).reshape(NC, B).astype(np.float64)
        hcell = np.bincount(cf.astype(np.int64) * B + b, weights=hf, minlength=NC * B).reshape(NC, B)
        msh = ncell[ncell > 0].mean()
        rc = (hcell + msh * rate[None, :]) / (ncell + msh)
        return dict(c=cb.tolist(), r=rc[:, keep].tolist(), pseudo=float(msh))

    def sse_apply(self, out, ctx, cal):
        c = np.asarray(cal["c"]); r = np.asarray(cal["r"])
        lq = np.log(np.maximum(out, 1e-12))
        o2 = np.empty_like(out)
        for cc in np.unique(ctx):
            mk = ctx == cc
            o2[mk] = np.interp(lq[mk], c, r[cc])
        o2 = np.maximum(o2, 1e-9)
        return o2 / o2.sum(1, keepdims=True)

    # ------------------------------------------------------------ ajuste
    def fit(self, train, V):
        self.V = V
        t_start = time.time()
        n = len(train)
        A = self._atoms_for(train)
        cut = int(n * (1 - self.dev_frac))
        tabs_cut = self._count(A, train, self.t0, cut)
        M = min(self.dev_positions, n - cut)
        dev = np.linspace(cut, n - 1, M).astype(np.int64)
        P64, LH, NN = self._chain_dists(tabs_cut, A, dev, V)
        del tabs_cut
        P = [p.astype(np.float32) for p in P64]
        del P64
        xd = train[dev]
        K = len(P)
        Gd = self.gates(A[dev], LH, NN)
        self.tabs = self._count(A, train, self.t0, n)     # tablas finales con todo el train (como v0.1)
        del A
        ia = np.arange(0, M // 3)
        ib = np.arange(M // 3, 2 * (M // 3))
        ic = np.arange(2 * (M // 3), M)
        self.idx = (ia, ib, ic)
        self.ptrue = np.stack([P[k][np.arange(M), xd] for k in range(K)], 1).astype(np.float64)
        self.xd = xd
        self.P, self.Gd = P, Gd
        self.log(f"[{self.name}] dev listo ({M} posiciones) t={time.time()-t_start:.1f}s")
        self._fit_heads()
        self.fit_time = time.time() - t_start
        self.log(f"[{self.name}] cabezas ajustadas en {self.fit_time:.1f}s")

    def _sub(self, idx):
        return [p[idx] for p in self.P], {k: v[idx] for k, v in self.Gd.items()}

    def _fit_heads(self):
        ia, ib, ic = self.idx
        xd = self.xd
        pt = self.ptrue
        K = pt.shape[1]
        PA, GA = self._sub(ia)
        xA, ptA = xd[ia], pt[ia]
        PB, GB = self._sub(ib)
        xB = xd[ib]
        PC, GC = self._sub(ic)
        xC = xd[ic]
        H = {}

        self._t0 = time.time()

        def add(name, head):
            H[name] = head
            self.log(f"   ajustada {name} (t={time.time()-self._t0:.0f}s)")

        # cabezas base de capa 1 (EM)
        Wb = self._em_w(ptA, GA["base"], self.GSIZE["base"])
        add("em_base", dict(kind="em", g="base", W=Wb))
        Wg1 = self._em_w(ptA, np.zeros(len(xA), np.int64), 1)
        add("em_global", dict(kind="em", g="base", W=np.repeat(Wg1, self.GSIZE["base"], 0)))
        # flat fine: EM sobre compuerta fina sin jerarquia (dispersion). G grande pero vectorizado.
        Wff = self._em_w(ptA, GA["fine"], self.GSIZE["fine"])
        add("em_flat_fine", dict(kind="em", g="fine", W=Wff))
        Wh = self.em_hier(ptA, GA["fine"], self.GSIZE["fine"], Wb, self.radix_fine_rest(), self.em_iters)
        add("hier_fine", dict(kind="em", g="fine", W=Wh, tau2=self.last_tau2, eb_mean=self.last_eb_strength_mean))
        # PoE
        LPA = np.log(np.maximum(np.stack([PA[k] for k in range(len(PA))]), 1e-12)).astype(np.float64)
        Wpg = self.poe_fit(LPA, xA, np.zeros(len(xA), np.int64), 1)
        add("poe_global", dict(kind="poe", g="base", W=np.repeat(Wpg, self.GSIZE["base"], 0)))
        if self.do_poe_base:   # PoE con compuerta: lento en numpy (ver NOTAS); desactivado en la corrida final
            Wpb = self.poe_fit(LPA, xA, GA["base"], self.GSIZE["base"])
            add("poe_base", dict(kind="poe", g="base", W=Wpb, poe=self.last_poe))
        # stack 2 capas: 5 mezclas de capa 1 con compuertas distintas; capa 2 sobre sus salidas (tramo dB)
        l1 = []
        for gname in ("base", "c1", "cls", "S", "L"):
            Wj = self._em_w(ptA, GA[gname], self.GSIZE[gname])
            l1.append((gname, Wj))
        qB = []
        for gname, Wj in l1:
            Wm = Wj[GB[gname]]
            q = np.zeros((len(xB), self.V))
            for k in range(K):
                q += Wm[:, k:k + 1] * PB[k]
            qB.append(q)
        y2 = np.stack([qq[np.arange(len(xB)), xB] for qq in qB], 1)
        W2 = self._em(y2, GB["base"], self.GSIZE["base"])
        add("stack2", dict(kind="stack", layer1=l1, g2="base", W2=W2))
        self.heads = H
        self.log(f"[{self.name}] cabezas: {list(H.keys())}  (t={time.time()-self._t0:.0f}s)")

        # evaluacion dev + SSE (SSE ajustada en dB, evaluada en dC)
        self.dev_report = {}
        for name, head in list(H.items()):
            outB = self.head_forward(head, PB, GB)
            cal = self.sse_fit(outB, xB, GB["cls"] // 5)
            H[name + "+sse"] = dict(kind="sse", base=head, cal=cal)
        for name, head in H.items():
            outC = self.head_forward(head, PC, GC)
            bpc = float(-np.log2(np.maximum(outC[np.arange(len(xC)), xC], 1e-12)).mean())
            self.dev_report[name] = bpc
            self.log(f"   dev-C bpc {name:18s} {bpc:.4f}")
        self.head = H["em_base"]

    def radix_fine_rest(self):
        # child = base * (RS*RL) + ... -> padre = child // (RS*RL)
        return self.radix["S"] * self.radix["L"]

    def n_params(self):
        return int(sum(len(t["psym"]) for t in self.tabs.values()))

    def predict(self, ids, pos):
        pos = np.asarray(pos, dtype=np.int64)
        A = self._atoms_for(ids)
        P, LH, NN = self._chain_dists(self.tabs, A, pos, self.V)
        G = self.gates(A[pos], LH, NN)
        out = self.head_forward(self.head, P, G)
        out = np.maximum(out, 1e-9)
        return (out / out.sum(1, keepdims=True)).astype(np.float32)


class Fixed:
    """Envoltorio: fit ya hecho (para harness.run sobre la cabeza elegida sin reajustar)."""
    def __init__(self, m, name):
        self.m, self.name = m, name

    def fit(self, train, V):
        pass

    def predict(self, ids, pos):
        return self.m.predict(ids, pos)

    def n_params(self):
        return self.m.n_params()

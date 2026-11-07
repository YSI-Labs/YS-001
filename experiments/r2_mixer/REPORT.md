# REPORT r2_mixer (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de mixer_ablation.json, final_poe_global.json, base_repro.json, mix.log)

**Veredicto: el producto de expertos (PoE, combinacion geometrica) con UN vector de pesos global gana -0.039 bpc: 2.0784 -> 2.0391.** La compuerta fina, el SSE y la mezcla de 2 capas no aportan o fallan.

## Idea
Sustituir p = sum_k w_k p_k por p(x) proporcional a prod_k p_k(x)^{w_k}. Pesos por maxima verosimilitud (funcion concava): Newton con Hessiana exacta (Cov_q(L) + lam I) y busqueda de linea Armijo; sin learning rate. Pruebas adicionales: compuerta fina jerarquica con shrinkage EB, mezcla en 2 capas (stack2), SSE por cuantiles. Dev interno del train en tercios (dA ajuste de pesos, dB capa 2 y SSE, dC seleccion); val no se uso para elegir.

## Resultados (val n_eval=20000; dev-C = held-out interno)
| cabeza | dev-C bpc | val bpc |
|---|---|---|
| em_base (= v0.1, pesos en dA) | 2.0302 | 2.0771 |
| em_global (EM sin compuerta) | 2.0248 | 2.0857 |
| em_flat_fine (EM compuerta fina plana) | 2.0277 | 2.0820 |
| hier_fine (fina con shrinkage EB) | 2.0302 | 2.0771 |
| **poe_global (PoE, sin compuerta)** | **1.9833** | **2.0391** |
| stack2 (2 capas EM) | 1.9879 | 2.0455 |
| em_base + SSE | 2.2736 | 2.3091 |
| poe_global + SSE | 2.2046 | 2.2486 |
| stack2 + SSE | 2.2370 | 2.2828 |
| poe_base (PoE con compuerta) | no completada (>20 min, abortada) | - |
Final (harness.run, leak_free=True, norm_err 4e-8): 2.0391 bpc. Base reproducida: 2.0784. Params: 15.65M entradas + pesos de mezcla (4 con PoE global).

## Hallazgos
- PoE global gana 0.047 bpc sobre la mezcla lineal global en val y en dev-C: la ganancia real de esta linea.
- La compuerta de la mezcla lineal aporta ~0.005-0.008 (confirma v0.1: la compuerta EM aporta poco).
- stack2 mejora sobre em_base (-0.032 val) pero queda detras de poe_global; no se probo stack2 sobre PoE.
- Compuerta fina con shrinkage EB: el estimador devolvio tau2 = 1e-8 (clamp) en las 4 coordenadas, colapsando a em_base; no es conclusion firme (posible sesgo del estimador: las responsabilidades no son Bernoulli).

## Fallas
1. PoE con compuerta: Newton por compuerta en numpy tardo >15 min; abortada (sin dato de bpc). Falta solver 4x4 vectorizado por lotes.
2. SSE: empeora todas las cabezas en +0.24 a +0.27 bpc (hipotesis NO verificada: bug de calibracion/abscisa en media geometrica). NO fusionar tal cual.
3. Bug de desarrollo: K indefinido en _fit_heads (corregido).
4. Mezcla en dA=50k (vs 150k en v0.1): reproduce 2.0771 vs 2.0784, el recorte no cambia el resultado de forma apreciable.

## Hiperparametros nuevos
6 numericas (24 bins SSE, Armijo c=1e-4 y factor 1/2, 8 halvings, 25 iteraciones Newton, tolerancia 1e-7, ridge 1e-6) + 3 clamps/tolerancias. En la cabeza ganadora (poe_global) solo las de Newton; sin compuerta, sin shrinkage. Los pesos globales NO se guardaron en JSON (conviene loguearlos al ajustar).

## Parche a fusionar (v0.2)
Reemplazar la combinacion lineal por PoE global: `poe_newton(LP, xs, g=zeros, G=1, w_init=1/K, w0=1/K, lam=0)` para obtener W0; predict: `L = sum_k W0[k]*log(max(P_k,1e-12)); out = softmax_V(L)`. Codigo de `poe_newton` en `mixer.py` (~45 lineas).
No fusionar: SSE, compuerta fina EB, stack2 sin revisar, PoE con compuerta.
Proximos: PoE con compuerta vectorizada; corregir SSE (abscisa en media aritmetica de q por bin); stack2 sobre PoE.

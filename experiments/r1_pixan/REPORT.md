# REPORT r1_pixan (transcrito por el orquestador desde el hand-back del agente Haiku; las cifras salen de result.json / sweep_dev.json de esta carpeta)

**Veredicto: NEGATIVO.** Estado fijo aleatorio + SimHash + tablas de conteo: 4.144 bpc (val) vs 2.269 de WB-5.

## Idea
K=9 integradores con fugas (tau = 2^0..2^8, a=exp(-1/tau)) sobre embedding gaussiano aleatorio por caracter (d=4 por escala, D=36), cuantizado con SimHash (b hiperplanos por tabla, T tablas). Tablas de conteo (bucket,char) con estimador Witten-Bell y fallback unigrama Laplace; mezcla de tablas por masa de conteo. Sin gradientes, numba. Ablacion `ExactCtx(k)` (clave = ultimos k chars, sin SimHash).
Fuera de alcance: estado no lineal (tanh/sign), automata celular, backoff jerarquico entre tablas, barrido de semilla.
Seleccion solo en dev (ultimo 5% del train). Config final: T=32, b=16, d=4, taus 2^0..2^8.

## Resultados (val, n_eval=50k; params = celdas (bucket,char) no nulas)
| variante | bpc val | params | fit_wall_s | train ch/s | eval ch/s |
|---|---|---|---|---|---|
| WB-ngram-5 (ref) | 2.269 | 1.05M | ~0.9 | - | - |
| ExactCtx k=4 (sin SimHash) | 2.337 | 0.29M | 2.9 | 1.65M | 36.9k |
| Pixan T=32 b=16 d=4 | 4.144 | 29.5M | 51.6 | 93k | 964 |

Dev (20k pos; ref WB-5 = 2.198): exacto k=1..5: 3.635/2.941/2.442/2.284/2.360.
Bits b (T=8): 10:4.461 12:4.359 14:4.361 16:4.316 18:4.239 20:4.228.
Tablas T (b=16): 1:4.627 2:4.445 4:4.343 8:4.316 16:4.249 32:4.213.
Dim d (b=16,T=8): 2:4.331 4:4.316 8:4.252 16:4.208.
Escalas (b=16,T=8,d=4): {1}:3.895 {1..4}:3.606 {1..16}:3.891 {1..256}:4.316 {16..256}:4.742 {256}:4.828.
Muestra: ilegible tras ~40 chars ("Hath h|e ntenetr r l se,O nihIyullla oiaest...").

## Hiperparametros
Libres de peso real: T, b, d, rango de escalas (4). Ademas: regla de mezcla, Laplace +1, semilla (no barrida). Las escalas potencia de 2 completas son PEORES que solo escalas cortas -> ninguna regla "natural" funciono.

## Fallas / hallazgos negativos
1. No supera a n-gram; ni siquiera tau=1 solo (3.89) iguala al exacto k=1 (3.63).
2. No es falta de datos: 0-0.07% de consultas caen en bucket vacio; los buckets (b=16, ~73 posiciones c/u) estan poblados pero mezclan contextos de siguiente-caracter distinto.
3. Las escalas largas danan (hipotesis NO verificada: tras estandarizar, la parte lenta domina la geometria de los hiperplanos y el bucket refleja historia lejana, no el caracter actual).
4. Incluso con tau=1, SimHash (d=4) no recupera la identidad del caracter (96 simbolos, pocas celdas utiles). No investigado por presupuesto.
5. T da rendimientos decrecientes (T=1 4.63 -> T=32 4.21) a costo lineal de memoria. Inferencia ~1k ch/s (T busquedas de V elementos por consulta).

## Escalabilidad
T: 1->32 celdas 0.92M->28.8M, fit 3.9s->47.4s. b: 16->20 con T=8 celdas 7.3M->17.8M (x2.4), fit 9.0->15.3s, bpc 4.316->4.228.

## Veredicto del agente
~3% de probabilidad de superar a WB-5 sin cambiar el diseno: el mecanismo de generalizacion no aparece (similitud en un estado aleatorio no se relaciona con el siguiente caracter). Siguientes pasos si se insistiera: (a) embeddings derivados de conteo/SVD de bigramas (cerrado), (b) SimHash sin escalas largas o backoff WB jerarquico, (c) ExactCtx k=4 + backoff como control. Si no bajan de ~2.5 bpc en dev, cerrar como negativa.

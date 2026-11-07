# REPORT r1_yaxche (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de result.json / scan.json / dev_*.json / run_log.txt de esta carpeta)

**Veredicto: NO supera a WB-5.** Mejor val = 2.390 bpc (B=64, 19.65M params, fit 421 s) vs WB-5 2.269 (1.08M params, 0.6 s).

## Idea
Arboles de contexto aprendidos (Yax Che', la ceiba). Preguntas "char(t-j) in S" (j=1..8, S = caracter o clase de 9 grupos) y "distancia al ultimo espacio/salto <= k". Hojas con conteos suavizados Witten-Bell (prior = hoja padre). Division codiciosa por ganancia KT (alpha=1/2) menos coste ln(Q); parada por MDL (la profundidad la decide el dato). Bosque de B arboles, cada uno con bootstrap de S=1M posiciones; promedio uniforme.

## Resultados (val, n_eval=50k)
| B | bpc | params | fit s | inferencia ch/s |
|---|---|---|---|---|
| 1 | 2.566 | 306k | 10.7 | 250.7k |
| 4 | 2.439 | 1.23M | 32.2 | 179.3k |
| 16 | 2.398 | 4.91M | 116.7 | 44.8k |
| 64 | 2.390 | 19.65M | 421.4 | 7.9k |
| WB-5 (ref) | 2.269 | 1.08M | 0.6 | |
B=64: leak_free=True, max_norm_err 4.5e-8. Profundidad media de hoja 17.7, max 44 (el tope no actua: cap 40 y 64 dan igual bpc en dev). Ablacion de profundidad (B=64, val 20k): d=2 4.22; 4 3.72; 6 3.37; 8 3.15; 12 2.82; 16 2.61; 24 2.42; 32 2.397; 64 2.396.
Muestra (B=64): "...DUKE.\nHath him I willes\nday the Deriorying have love, morn of those it all of by any withoute, this,\nToo truesting. Forthum say watch, leave forged thy live..."
Dev (ultimo 5% train): S=200k B=1 2.858; S=500k B=4 2.518; S=1M B=4 2.403. feat_frac=0.5 -> 2.94 (empeora). L=8 empata con L=16 (2.516 vs 2.518); se elige L=8.

## Hiperparametros
Libres reales ~5-6: L=8, S=1M, B, DMAX=31, lista de clases (9 grupos a mano), umbral de espacio. Teoricos/topes: KT alpha=1/2, prior WB (sin constante), coste ln(Q), max_depth 64, maxnodes 300k, seed 0.

## Fallas / hallazgos negativos
1. Subespacio aleatorio de features por arbol (estilo random forest): muy malo (2.94 vs 2.52); los lags cercanos son esenciales.
2. Mas arboles casi no ayudan: 16->64 solo -0.008 bpc por 3.6x tiempo y 4x params. El cuello es el sesgo de cada arbol.
3. Mas datos por arbol importan mas que mas arboles (500k->1M con B=4: -0.115; 4->16 arboles: -0.04).
4. Division codiciosa sin lookahead puede ocultar interacciones tipo XOR (no medido).
5. Incoherencia: la parada usa codelength KT, la prediccion usa Witten-Bell.
6. A ~1M params el bosque pierde con WB-5 (B=1: 306k params, 2.566).

## Escalabilidad
Params y fit ~lineales en B; inferencia ~1/B. bpc: B 1->4 -0.126; 4->16 -0.041; 16->64 -0.008. Memoria no medida.

## Veredicto del agente
~8%. El mejor punto es 0.12 bpc peor que WB-5 con 18x params y ~700x tiempo de fit; la estructura de contexto aprendida no mejora el contexto contiguo. Siguientes pasos: Bayes por evidencia en lugar de promedio uniforme; lookahead/division multinivel; unificar codelength; solo si cierra el gap, escalar S.
Limites: L y S elegidos con dev; B=1,4,16,64 son prefijos del mismo bosque de 64 arboles; fit medido con 2 hilos y equipo compartido.

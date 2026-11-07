# REPORT r2_cache (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/r2_*.json)

**Veredicto: la memoria EN CONTEXTO (cache adaptativo) es el mayor avance hasta ahora: 2.0784 -> 1.9615 bpc (-0.117) sin entrenar nada.** El match model no aporta encima del cache.

## Idea
Dos expertos sin parametros aprendidos que solo miran `ids[:pos]` (causales): (1) `CacheWB`: Witten-Bell jerarquico (orden 4, base unigrama) con conteos exactos dentro de ventanas 256/1024/4096/todo el historial; (2) match model contra historial y contra el train (hash de 6 y 12 chars, verificacion hacia atras hasta 32; confianza por conteo calibrada en dev). Se integran como columnas nuevas de la mezcla EM de Ixchel (nucleo generalizado en `core.py` de esta carpeta; con `cache_windows=()` y `match_sources=()` reproduce v0.1 exactamente: 2.078358).

## Resultados (val, n_eval=20000, leak_free=True)
| variante | bpc val | delta vs v0.1 | celdas | fit s (contaminado) | inf ch/s | bpc dev mezcla | copia media (3000 ch) |
|---|---|---|---|---|---|---|---|
| base v0.1 (A+W+S+L) | 2.0784 | 0 | 15.65M | 100.7 | 25,184 | 1.9878 | 7.11 |
| + cache (4 ventanas) | 1.9615 | -0.117 | 15.65M | 107.1 | 8,163 | 1.8582 | 4.82 |
| + match (hist+train) | 2.0473 | -0.031 | +14 | 113.0 | 6,150 | 1.9483 | 5.65 |
| + cache + match | 1.9614 | -0.117 | +14 | 118.8 | 4,438 | 1.8581 | 4.41 |
| + cache + match, compuerta extendida | **1.9575** | -0.121 | +14 | 108.5 | 5,409 | 1.8230 | 5.91 |
Referencias: WB-5 2.269; GRU 0.84M (8 min) 2.314. Copia literal de referencia (texto real val) 10.20 de media; ninguna variante copia mas que v0.1.
Cache por ventana, dev, expertos solos: C256 4.28; C1024 3.70; C4096 3.24; Cinf 2.44 bpc.
Muestra (mejor): "...DUKE.\nHath he a storms.\nT l,elbl EnteryurninWimampull'd look upon her.\n\nEDGAR.\n[_. nit: lyongdoollow put..." (ruido ortografico con fragmentos de estilo; sin coherencia de frase).

## Hallazgos
1. El cache da casi toda la ganancia (-0.117 val, -0.13 dev): nombres de personaje y frases repetidas dentro del texto.
2. El match no aporta sobre el cache (1.9614 vs 1.9615); solo ayuda sin cache (-0.031). Calibracion correcta (acierto 0.57 con 6-7 chars -> 0.91 con >=32) pero sin beneficio marginal.
3. Compuerta extendida: -0.004 val pero -0.035 dev (desfase x9): sobreajuste leve; marginal.
4. NO medido: ablacion de ventana individual (solo el conjunto de 4); control sin repeticiones (texto barajado por bloques).

## SALVEDAD METODOLOGICA (orquestador)
El cache usa TODO el historial de `val` anterior a la posicion evaluada (hasta 536k caracteres = varias obras), mientras la GRU/transformer de referencia ven solo 128 caracteres. Es legitimo (causal, leak_free) y es "aprendizaje en contexto" real, pero la comparacion con las redes de referencia no es de contexto igual, y la ganancia depende de cuanto se repite el texto (nombres de personajes recurrentes) en el documento evaluado. Hay que reportar ambos: Ixchel sin cache (2.078) y con cache (1.96), y repetir la medicion con ventana acotada y con barajado por bloques.

## Hiperparametros (todos fijados a priori; ninguno ajustado con val)
Orden maximo del cache 4; ventanas {256,1024,4096,inf}; longitud minima del match 6, tope de verificacion 32, clave larga 12; bordes de bucket de longitud {6,8,12,16,24,32}; gate extra binarios; prior_m=1; TBITS=27 (estructural). ~7 decisiones de diseno, cero constantes por gradiente.

## Costes y escala
Cache sin fit; match anade 2 argsort de 4.8M (~+7 s). `predict` reconstruye el indice de `ids[:maxpos]` en cada llamada (O(N log N)): evaluacion 4-8k ch/s y generacion O(L^2 log L); para produccion hace falta indice incremental. Memoria ~64 B/char de historial (cache), ~150 MB indice del match en train completo.

## Recomendacion de fusion para v0.2
Si: `CacheWB` (`cache_experts.py`), minimo ventanas (1024, inf). No (por ahora): match model. Opcional: compuerta extendida (riesgo de sobreajuste). Patch de mezcla en `core.py` desde la clase `Ixchel` (`_expert_dists`, `_combine`, `_ctx_prepare`, `fit` con dev por chunks); quitar el caso especial de ROOT al fusionar. Proximos: ablacion de ventanas, control sin repeticiones, indice incremental.

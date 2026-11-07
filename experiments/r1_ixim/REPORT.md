# REPORT r1_ixim (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/R0_L32_D2048.json y logs)

**Veredicto: NEGATIVO (en esta forma).** Capa oculta aleatoria fija + salida ridge cerrada: 4.334 bpc (val, n_eval=20k) vs 2.269 de WB-5; apenas mejor que el unigrama (~4.7).

## Idea e implementacion
Contexto = ultimas L=32 letras; cada (posicion,caracter) tiene un vector gaussiano fijo en R^256 que se suma; ReLU de una proyeccion aleatoria fija a D=2048; salida = ridge en forma cerrada (Gram acumulado por bloques, eigh, lambda por GCV), con calibracion a distribucion (clip / softmax-1D / Newton) elegida por NLL en dev. Sin backprop ni learning rate. Control `ngram` (one-hot de n-gramas K=4, D=2048) implementado pero SIN resultado al cierre del informe.
Fuera de alcance: multiples capas, D=4096/8192, varias semillas, corrida final n_eval=50k.

## Resultados (val, n_eval=20k)
| config | calibracion | bpc val | params aprendidos | fijos aleatorios | fit s | train ch/s | inf ch/s |
|---|---|---|---|---|---|---|---|
| R0 L32 D2048 d0=256 | clip (elegida en dev) | 4.3339 | 196,801 | 1,318,912 | 614.8 | 7,847 | 25,907 |
| R0 | newton | 4.3578 | | | | | 23,548 |
| R0 | softmax-1D | 4.9855 | | | | | 23,374 |
| smoke L16 D256 (solo consola) | clip | 4.4084 | 24,769 | | 46.2 | | |
Muestra: ilegible, caracteres basura; no es lenguaje sostenido. leak_free=True.

## Hiperparametros
Libres reales ~5 (L, D, d0, seed, K) elegidos a mano; ademas rejillas (lambda 41 pts, s 41 pts, eps 33 pts), DEV_FRAC=0.05, CHUNK=8192, Newton 8 iter. Derivados de datos: lambda (GCV: rel 0.00562; cross-check dev: 0.00316) y calibracion.

## Fallas / hallazgos negativos
1. No pasa del nivel unigrama. Hipotesis (NO comprobada; no se midio d0 mayor): sumar 32 vectores aleatorios en 256 dims produce interferencia y pierde la identidad del contexto.
2. La regresion sobre one-hot no da probabilidades; softmax-1D peor (4.99), clip mejor, Newton cercano.
3. Coste: Gram N*D^2: 615 s para D=2048 con 4.56M ejemplos. D=4096 ~40 min y D=8192 ~2.5 h (extrapolado, no medido). Lineal-sobre-rasgos-densos no escala en CPU en comparacion con contar.
4. Metodo: fit con train[:95%] (4.56M); dev fuera para calibrar.

## Escalabilidad
Solo 2 puntos (D=256 L=16: 46 s; D=2048 L=32: 615 s), sin curva completa.

## Veredicto del agente
~3%. Cuello de botella = la representacion del contexto, no la lectura lineal. Siguiente paso: sustituir sumas por n-gramas hasheados 1..5 con la misma lectura cerrada (control N0); si N0 tambien queda ~4 bpc, descartar la familia lineal sobre rasgos fijos.

## ADENDA del orquestador (resultado posterior al hand-back: results/N0_K4_D2048.json)
Control N0 (one-hot de n-gramas de orden <=4, D=2048, MISMA lectura ridge cerrada): **2.630 bpc val (n_eval=20k), fit 351 s**, vs 4.334 de los rasgos aleatorios (R0) y 2.269 de WB-5 (conteo directo).
Conclusion: la hipotesis del agente se confirma: el cuello de botella era la REPRESENTACION (sumar vectores aleatorios destruye la identidad del contexto), no la lectura lineal cerrada. Pero incluso con buena representacion, ridge sobre n-gramas es 0.36 bpc peor que contar y 100x mas lento (351 s vs 0.6 s): la solucion cerrada densa no compite con el conteo disperso.

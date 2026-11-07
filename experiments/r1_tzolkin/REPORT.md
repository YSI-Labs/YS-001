# REPORT r1_tzolkin (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de grid/*.json, result.json y dev_*.json de esta carpeta)

**Veredicto: la idea de mascaras aleatorias dispersas NO funciona.** El unico buen resultado (2.214 bpc) es el ensamble CONTIGUO exacto, que en esencia es un ensamble de n-gramas de ordenes 1..8 con 14x mas celdas que WB-5 (no es un descubrimiento).

## Idea
M tablas de conteo, cada una indexada por una mascara aleatoria de las ultimas L=16 posiciones (offset 1 siempre, k en 1..8). Entrenamiento = conteo. Cada tabla da Witten-Bell con base unigrama; combinacion ponderada por evidencia (geometrica o lineal). Variantes: hash denso 2^b o indice exacto.

## Resultados (val, n_eval=50k; leak_free=True en todas)
| variante | bpc | params alloc | celdas no nulas | fit s | inf ch/s |
|---|---|---|---|---|---|
| Contiguo exacto M=16 (geo_seen) | 2.214 | 14.7M | 14.7M | 17.5 | 2277 |
| Contiguo denso M=16 b=16 | 2.645 | 100.7M | 9.5M | 19.3 | 7511 |
| Aleatorio exacto M=16 | 2.940 | 21.7M | 21.7M | 16.9 | 2238 |
| Aleatorio denso M=16 b=18 | 3.181 | 402.7M | 16.6M | 20.7 | 5771 |
| Aleatorio denso M=16 b=16 | 3.259 | 100.7M | 10.4M | 16.7 | 5986 |
| Aleatorio denso M=16 b=14 | 3.374 | 25.2M | 4.9M | 16.3 | 6489 |
| Aleatorio denso M=64 b=16 (headline) | 3.450 | 402.7M | 55.9M | 67.9 | 1759 |
| Aleatorio denso M=256 b=14 | 3.612 | 402.7M | 111.9M | 208.6 | 724 |
Ref WB-5 = 2.269 (1.08M). Tiempos contaminados (2 hilos, CPU compartida).
Regla de mezcla elegida en dev (M=16 aleatorio exacto, bpc dev): lin_wb 3.012, geo_wb 2.939, lin_T 3.643, geo_logT 3.144, geo_seen 2.922 (elegida). Dev: aleatorio 2.922 vs contiguo 2.174.
Muestra (contiguo exacto M=16): "...DUKE.\nHath he PATherent\ns ses, had refyring and low lard. He that hast all percme preuntoxter."

## Hiperparametros
Libres: M, L=16, kmax=8, sesgo offset-1, prior 1/o de offsets, b, semilla 0; mas una eleccion discreta de regla en dev y fraccion dev 5%. 7 libres estructurales + 1 eleccion en dev. **No es "sin hiperparametros".**

## Fallas / hallazgos negativos
- Mascaras aleatorias dispersas (offsets ~1/o desde 2..16): 2.94 vs 2.21 contiguo (val); 2.92 vs 2.17 (dev). La hipotesis de skip-gramas utiles no se sostiene en esta forma.
- Colisiones del hash denso cuestan ~0.43 bpc incluso en contiguo (denso b=16 2.65 vs exacto 2.21); aleatorio: exacto 2.94, b=18 3.18, b=16 3.26, b=14 3.37.
- Mas tablas EMPEORAN con la mezcla actual (dev, b=16: M=16 3.38 lin / 3.22 geo; M=64 3.55 / 3.40): dilucion, el producto geometrico y la mezcla promedian tablas casi-unigrama con las informativas; la ponderacion T/(T+U) no discrimina (w~1 en tablas colisionadas). El problema es la combinacion, no solo las tablas.
- 2.21 no es comparacion a igual parametros: WB-5 usa backoff jerarquico y 14x menos celdas.
- Bug detectado: run.py calculaba n_nonzero antes del fit; corregido antes de las corridas finales.

## Escalabilidad
Fit ~lineal en M (M=16 17 s, 64 68 s, 256 209 s); inferencia ~lineal en M (2238/1759/724 ch/s). Memoria densa buckets*V*M. No se midio bpc vs tamano del corpus.

## Veredicto del agente
~15% para la familia (tablas de conteo con mascaras) de superar a WB y acercarse a redes pequenas; <5% la version aleatoria. Siguientes pasos: backoff jerarquico WB sobre ordenes contiguos a igual celdas; tablas exactas dispersas en lugar de hash denso; M contiguo con varias semillas; solo despues, mascaras salteadas con pesos de dev.

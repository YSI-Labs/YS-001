# Ronda 1 - resumen comparativo (val bpc; menor es mejor)

Corpus: Shakespeare completo (Gutenberg pg100), 4.82M chars train / 0.54M val, V=96. Azar = 6.58 bpc. Fuente de cada cifra: REPORT.md y JSON de cada carpeta r1_*.
Nota: n_eval varia (20k/50k) y los tiempos estan contaminados (8 agentes en 16 hilos); comparar ordenes de magnitud, no decimales.

| Candidato (nombre maya) | Mejor bpc val | Params/celdas | Fit (s) | Veredicto |
|---|---|---|---|---|
| WB-5 n-gram (baseline) | 2.269 | 1.08M | 0.6 | referencia |
| **Tz'ib** (KN cerrado, orden 5) | **2.185** (2.195 en 50k) | 0.41M | ~40 (indice) | Unico que mejora sin hiperparametros ajustados; orden infinito memoriza (WB-inf 3.12) |
| Tzolk'in (contiguo exacto M=16) | 2.214 | 14.7M | 17.5 | = ensamble de n-gramas 1..8; mascaras aleatorias 2.94 (fallo) |
| Ik' (SVD + WB-5, mezcla) | 2.213 | 17.9M | 56 | Ganancia casi toda de la mezcla; espectral puro 3.20 |
| Yax Che' (bosque MDL B=64) | 2.390 | 19.65M | 421 | Pierde con WB-5; mas arboles casi no ayudan |
| Ixim N0 (ridge sobre n-gramas, K=4) | 2.630 | 0.2M | 351 | Representacion exacta rescata la lectura lineal, pero 100x mas lenta que contar |
| Chan (HDC + WB, H=31) | 3.870 | 30.5M | 86 | Sketch aleatorio de tabla de conteo; ruido ~ sqrt(N/(D*H)) |
| Pixan (reservorio + SimHash) | 4.144 | 29.5M | 52 | Buckets mezclan contextos con siguiente-char distinto |
| Ixim R0 (rasgos aleatorios + ridge) | 4.334 | 0.2M | 615 | Interferencia destruye identidad del contexto |
| baseline neuronal (torch) | (pendiente) | | | referencia de calidad por CPU-segundo |

## Lecciones transversales (para el preprint)
1. Todo estado denso/aleatorio que comprime el contexto (Pixan, Ixim-R, Chan) pierde la identidad exacta del contexto y queda 1.5-2 bpc por detras del conteo exacto.
2. El conteo exacto + suavizado correcto domina el rango 2.2-2.3 bpc en 5 MB: el techo de "contexto local exacto" esta en ~2.2.
3. El orden ilimitado sin suavizado cuidadoso memoriza (Tz'ib WB-inf: 100% de la salida copiada del train).
4. Mezclar muchos expertos sin pesos informados DILUYE (Tzolk'in M=64 peor que M=16); la combinacion es el punto debil, no las tablas.
5. Mas arboles/tablas/dimensiones rinden rendimientos decrecientes; mas DATOS rinden de forma estable (~-0.08 bpc por duplicar en KN-5).
6. La SVD aporta senal sobre proyeccion aleatoria (3.37 vs 4.31) pero poca en mezcla (0.04 bpc).

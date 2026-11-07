# Ixchel en text8 - escalera de datos (test estandar: ultimos 5M de text8; n_eval=50000; leak_free verificado en subconjunto de 300k)

| config | train chars | bpc | pares | fit s (pared) | fit CPU-s | RAM pico MB |
|---|---|---|---|---|---|---|
| A | 1,000,000 | 2.0421 | 1,585,582 | 7 | 8 | 527 |
| A | 3,000,000 | 1.8990 | 3,548,083 | 10 | 13 | 961 |
| A | 10,000,000 | 1.7597 | 8,109,350 | 34 | 38 | 2489 |
| A | 30,000,000 | 1.6684 | 16,201,576 | 109 | 119 | 6595 |
| full | 1,000,000 | 1.7664 | 4,569,328 | 12 | 19 | 698 |
| full | 3,000,000 | 1.6649 | 10,809,439 | 46 | 56 | 1470 |
| full | 10,000,000 | 1.5587 | 27,102,559 | 126 | 143 | 3869 |
| full | 30,000,000 | 1.4856 | 62,198,039 | 370 | 403 | 10341 |
| full | 90,000,000 (train estandar completo) | **1.4193** | 142,934,040 | 1015 | 1091 | 15012 |

A: pendiente por duplicacion de datos entre puntos consecutivos: -0.090, -0.080, -0.058
  ajuste a*N^-b+c (N en M chars, 4 puntos, 3 parametros): a=0.81, b=0.182 (se 0.043), c=1.23 (se 0.14); extrapolacion a 90M: 1.587, 900M: 1.464  (ORIENTATIVO: 4 puntos, 3 parametros)

full: pendiente por duplicacion de datos entre puntos consecutivos: -0.064, -0.061, -0.046
  ajuste a*N^-b+c (N en M chars, 4 puntos, 3 parametros): a=0.78, b=0.132 (se 0.048), c=0.98 (se 0.23); extrapolacion a 90M: 1.417, 900M: 1.303  (ORIENTATIVO: 4 puntos, 3 parametros)

Referencias publicadas text8 (entrenadas con los 90M, agregadores de resultados, no verificadas contra articulos originales): Transformer-XL 277M params ~1.08; mLSTM ~1.27; HM-LSTM ~1.29. Ixchel (full, v0.3 sin cadena L) con los 90M completos: 1.4193 bpc, 143M pares, 1015 s de entrenamiento, 15 GB RAM pico (tras refactor de memoria); la extrapolacion del ajuste con 1-30M daba 1.417: la curva de escala predijo el punto. Sigue ~0.14 bpc por encima de la mLSTM publicada (~1.27) y ~0.34 por encima de Transformer-XL (~1.08).

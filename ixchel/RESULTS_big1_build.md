# Ixchel grande (big1): construccion fuera de memoria

Corpus: 690.6M caracteres (web C4 69.5%, codigo Python 27.5%, dolly 1.7%, GSM8K 0.6%, Shakespeare 0.7%); tablas construidas sobre los primeros 656.1M (95%); el 5% final queda como dev.
Metodo: `ixchel/bigbuild.py` (escaneo en trozos de 12M caracteres con volcado a disco + reduccion por 32 particiones de rango de clave); 3 procesos en paralelo (4 hilos numba c/u) sobre una Ryzen 7 7730U (8 nucleos/16 hilos), maquina parcialmente compartida con otras tareas durante parte del escaneo.

| cadena | niveles | pares | contextos | tamano en disco |
|---|---|---|---|---|
| W | 4 | 410,742,544 | 335,253,292 | 6.91 GB |
| A | 9 | 244,465,648 | 144,323,977 | 3.29 GB |
| W2 | 3 | 129,270,032 | 97,602,735 | 2.06 GB |
| Kc | 4 | 55,728,340 | 26,314,992 | 0.65 GB |
| S | 5 | 39,591,647 | 15,310,961 | 0.41 GB |
| L | 4 | 23,103,601 | 5,181,428 | 0.18 GB |
| Kb | 4 | 10,007,108 | 2,711,868 | 0.09 GB |
| WL | 3 | 751,478 | 40,114 | 0.00 GB |

**Total: 913,660,398 pares de conteo (0.914 mil millones) en 13.6 GB (16.0 bytes por par).**

## Tiempos de construccion (pared)
| grupo de cadenas | escaneo (s) | reduccion (s) | total (s) |
|---|---|---|---|
| A, W2, WL | 1885 | 422 | 2306 |
| Kb, Kc | 1059 | 124 | 1183 |
| S, W, L | 1682 | 487 | 2169 |

Tiempo de pared total (el grupo mas lento): 2306 s = 38 min. Procesos-segundo sumados: 5658 s (cota superior de 6.3 CPU-horas con 4 hilos por proceso). Velocidad efectiva: 285 k caracteres/s de corpus procesado, 0.40 M pares/s.

Aproximaciones declaradas: los atomos de palabra/linea se reinician en cada trozo de 12M caracteres y se omiten los primeros 16 caracteres de cada trozo; en la deduplicacion de pares hijo para los conteos de continuacion se usan 32 bits altos del hash (verificado: tablas identicas a las del metodo en memoria en un corpus de 1.5M caracteres).

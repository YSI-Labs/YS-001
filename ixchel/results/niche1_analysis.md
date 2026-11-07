## E1: AUC (limpio vs degradado; 0.5 = azar) con IC95% bootstrap por documento; entre parentesis, precision pareada (mismo doc)
| degradacion | Ixchel completo | n-grama KN (A) | zlib |
|---|---|---|---|
| noise2 | 0.940 [0.926, 0.953] (1.00) | 0.950 [0.937, 0.962] (1.00) | 0.635 [0.626, 0.646] (1.00) |
| noise10 | 0.998 [0.994, 1.000] (1.00) | 0.997 [0.992, 1.000] (1.00) | 0.934 [0.919, 0.948] (1.00) |
| wshuf | 0.974 [0.963, 0.984] (1.00) | 0.984 [0.974, 0.991] (1.00) | 0.623 [0.613, 0.635] (0.95) |
| repeat | 0.000 [0.000, 0.000] (0.00) | 0.796 [0.772, 0.819] (0.86) | 0.000 [0.000, 0.000] (0.00) |
| splice | 0.629 [0.604, 0.653] (0.66) | 0.555 [0.528, 0.581] (0.55) | 0.701 [0.678, 0.724] (0.79) |

## E2: Spearman con el bpc de Qwen2.5-1.5B (n=200 docs C4 limpios), IC95% bootstrap; solape de los 20% peores
| puntuador | Spearman | IC95% | solape top-20% peor |
|---|---|---|---|
| Ixchel completo | 0.687 | [0.593, 0.764] | 0.60 (azar 0.20) |
| n-grama KN (A) | 0.592 | [0.480, 0.688] | 0.57 (azar 0.20) |
| zlib | 0.437 | [0.302, 0.559] | 0.40 (azar 0.20) |

bpc medio (limpio): Ixchel completo 1.699, n-grama KN (A) 1.808, zlib 0.540
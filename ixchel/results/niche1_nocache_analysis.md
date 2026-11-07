## Seguimiento (a): AUC limpio vs degradado con IC95% bootstrap por documento
| degradacion | Ixchel con cache | Ixchel sin cache | KN (A) | gap (repeticion) | combinado max-rango |
|---|---|---|---|---|---|
| noise2 | 0.940 [0.926, 0.953] | 0.943 [0.928, 0.956] | 0.950 [0.936, 0.964] | 0.645 [0.632, 0.659] | 0.796 [0.783, 0.811] |
| noise10 | 0.998 [0.994, 1.000] | 0.996 [0.990, 1.000] | 0.997 [0.991, 1.000] | 0.879 [0.858, 0.900] | 0.926 [0.908, 0.943] |
| wshuf | 0.974 [0.964, 0.983] | 0.978 [0.967, 0.987] | 0.984 [0.974, 0.991] | 0.706 [0.684, 0.726] | 0.841 [0.826, 0.855] |
| repeat | 0.000 [0.000, 0.000] | 0.806 [0.784, 0.827] | 0.796 [0.774, 0.817] | 1.000 [1.000, 1.000] | 0.916 [0.903, 0.927] |
| splice | 0.629 [0.605, 0.655] | 0.551 [0.526, 0.579] | 0.555 [0.530, 0.579] | 0.319 [0.296, 0.344] | 0.401 [0.376, 0.427] |

## E2: Spearman con bpc de Qwen2.5-1.5B (n=200), IC95% bootstrap
| puntuador | Spearman | IC95% |
|---|---|---|
| Ixchel con cache | 0.687 | [0.591, 0.762] |
| Ixchel sin cache | 0.612 | [0.500, 0.702] |
| KN (A) | 0.592 | [0.476, 0.692] |

bpc medio limpio: Ixchel con cache 1.699, Ixchel sin cache 1.695, KN (A) 1.808, gap (repeticion) -0.004
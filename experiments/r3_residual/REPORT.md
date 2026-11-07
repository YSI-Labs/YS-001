# REPORT r3_residual (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de result_*.json, base_*.json)
**Veredicto: el residuo neuronal pequeno mejora a Ixchel v0.3 pero NO iguala al ensamble Ixchel+GRU, y sigue dependiendo de AdamW. No se fusiona.**
Idea: p(x|ctx) proporcional a exp(a*log p_ixchel + b + f_theta(ctx)); f_theta = GRU 1 capa d=64 (37,344 params); entrenamiento SIN FUGA con validacion cruzada por bloques (K=4: tablas de 3 bloques predicen el 4o; 1.5M posiciones OOF, MVP); a,b por Newton en dev; AdamW lr 3e-3 coseno 2600 pasos (etiquetado), batch 64 x ctx 128.
## Resultados (val n_eval=10000, mismas posiciones que la GRU; SE bloque 500 / iid)
| modelo | params entrenados | CPU-s | bpc val | efecto pareado vs base | vs ensamble |
|---|---|---|---|---|---|
| Ixchel v0.3 (base) | 0 (20.1M celdas) | 108 | 1.9233 | - | ensamble mejor por 0.0398 |
| Ensamble Ixchel+GRU 0.84M (reproducido) | 0.84M | 1044 | 1.8835 | - | - |
| Residuo a+b | 37,344 | 822 | 1.8968 | mejora 0.0265 (SE .0202/.0076) | peor 0.0133 (SE .0193/.0077) |
| Residuo a escalar (post hoc) | 37,344+1 | 822 | 1.8912 | mejora 0.0321 (SE .0167/.0063) | peor 0.0076 (SE .0159/.0064) |
Dev: base 1.8113; residuo a+b 1.7476; a escalar 1.7510.
Verificacion de fuga: base OOF 1.8714 (folds 1.887/1.871/1.897/1.831) vs dev 1.8113 (consistentes); control con tablas de todo el train sobre train: 1.2569 bpc (fuga de 0.61 bpc que el OOF evita); sin sobreajuste de f (loss 1.793 vs dev 1.752).
## Fallas
1) No iguala al ensamble (diferencia 0.008 con a escalar: no significativa). 2) d=128 abortado por lento (0.6 CPU-s/paso). 3) El vector b sin regularizar daña (33 simbolos con <20 casos, b hasta -10.8; 22 posiciones raras costaron 18.3 bits): explica 0.0056 bpc. 4) Discrepancia dev/val de la mejora (0.06 vs 0.03): OOF usa tablas del 71% del train y el test el 100% (confianza del base desajustada). 5) Alcance reducido: 1.5M de 4.56M posiciones OOF, una semilla.
## Hiperparametros: ~11 libres nuevas (d, capas, ctx, batch, pasos, lr, warmup, wd, betas, clip, semilla) + K=4, M=375k, CTX_MIN=64; 6 dependen del optimizador (no cumple el objetivo sin LR).
Probabilidad estimada (agente) de que mas pasos superen al ensamble: 25-30%. Si se retoma: K=8, d>=96, a escalar en vez de b.

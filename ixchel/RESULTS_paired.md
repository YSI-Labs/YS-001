# Test pareado Ixchel vs GRU y ensamble (Shakespeare; mismas 10,000 posiciones de val; ixchel/results/ensemble_test.json)
GRU reproducible (experiments/r3_gru_paired): 0.84M params, AdamW lr 3e-3 coseno, batch 64 x ctx 128, 2600 pasos (21.3M tokens, 4.6 epocas), semilla 0, pesos guardados; val 10k = 2.0905 bpc (la corrida anterior por tiempo dio 2.0702; variacion entre corridas ~0.02, del orden del SE).
SE por bloques = desviacion de medias de bloques de 500 posiciones contiguas / sqrt(n bloques) (conservador por la correlacion local); los pareados comparan posicion a posicion.
| comparacion | diferencia (bpc) | SE iid | SE bloques |
|---|---|---|---|
| GRU - Ixchel v0.3 ctx 128 (mismo contexto) | +0.057 (Ixchel mejor) | 0.011 | 0.014 |
| GRU - Ixchel v0.3 cache largo | +0.167 | 0.013 | 0.018 |
| Ixchel(ctx128) - Ensamble(Ixchel ctx128 + GRU) | +0.064 | 0.005 | 0.006 |
| Ixchel(cache largo) - Ensamble(Ixchel + GRU) | +0.040 | 0.004 | 0.005 |
Niveles absolutos (SE de la media por bloques ~0.036-0.041): Ixchel ctx128 2.034; Ixchel cache largo 1.923; GRU 2.091; Ensamble ctx128 1.970 (pesos PoE Ixchel .585, GRU .439); Ensamble cache largo 1.884 (pesos .708, .330).
Dev (50k posiciones, 5% final del train, no visto por la GRU ni en los pesos del ensamble): Ixchel 1.934/1.806 (ctx128/cache), GRU 1.968.
Fraccion de posiciones donde Ixchel supera a la GRU: 51.3% (ctx128), 55.7% (cache largo).
Lectura: con contexto igualado Ixchel supera a la GRU de forma significativa (~4 SE) pese a ~100x menos CPU; la GRU aporta informacion complementaria (el ensamble mejora a ambos de forma significativa). Salvedades: una semilla de GRU, un tamano (0.84M), ~4.6 epocas; la GRU no esta en su optimo y una red mayor/mas entrenada reduciria la ventaja o la invertiria; un solo corpus.

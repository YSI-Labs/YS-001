# Ixchel v0.2 - resultados (val, n_eval=20000, leak_free=True; JSON en ixchel/results/v02_*.json)
v0.2 = v0.1 (cadenas A,W,S,L KN modificado) + cache en contexto (CacheWB, r2_cache) + mezcla geometrica global por Newton (r2_mixer). Tablas contadas con todo el train; pesos en dev interno (5% final del train).

| config | bpc val | dev bpc | params (pares) | fit s* | nota |
|---|---|---|---|---|---|
| v0.1 A+W+S+L, EM con compuerta | 2.0784 | 1.9878 | 15.65M | ~52 | base |
| poe (sin cache) | 2.0388 | 1.9617 | 15.65M | 48.6 | mezcla geometrica: -0.040 |
| cache128_poe (ventana 128 = ctx de la GRU) | 2.0288 | | 15.65M | 47.1 | contexto igualado a la GRU |
| cacheB_poe (ventanas 256,1024) | 1.9971 | | 15.65M | 50.9 | |
| cache2_poe (ventanas 1024, inf) | 1.9307 | | 15.65M | 48.7 | |
| cache_em (4 ventanas, mezcla lineal EM) | 1.9615 | 1.8582 | 15.65M | 42.0 | reproduce r2_cache |
| **cache_poe (256,1024,4096,inf)** | **1.9247** | 1.8215 | 15.65M | 56.5 | **mejor v0.2** |
*fit con la maquina compartida (otros agentes en marcha); CPU-s ~ 55-65.

Pesos PoE (cache_poe): A .453, W .209, S .079, L -.081, C256 .036, C1024 .062, C4096 .060, Cinf .196.
Referencias: WB-5 2.269 | GRU 0.84M 8 min 2.314 | GRU 0.84M 30 min (8 hilos, 11,381 CPU-s) 2.070 (n_eval=10k).
Lectura: con el MISMO contexto que la GRU (128 chars), Ixchel v0.2 da 2.029 vs 2.070 de la GRU. El cache de largo alcance aporta ~0.10 bpc adicionales, creciendo con la ventana: es memoria en contexto (nombres y frases repetidos), no aprendizaje de pesos.

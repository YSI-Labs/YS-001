# REPORT r3_engine (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de verify.json, bench.json, bench_conc.json)
Motor de inferencia INCREMENTAL exacto para Ixchel v0.3 (Shakespeare, config v03). `engine.py` (`IxchelEngine`: load, new_session, step, generate, generate_fast, feed, score_positions); depende solo de numpy y numba. Modelo en `model_v03/` (arrays .npy mmap, 422 MB incl. 128 MB de indice hash, carga 0.02 s). Atomos de palabra/linea por caracter; cache WB por ventanas como hash con +1 al entrar y -1 al salir de cada ventana finita; tablas de cadenas fijas.
## Equivalencia con Ixchel.predict (verify.json)
A) 200 posiciones de val: diferencia maxima 0.0 (bit a bit float32), argmax 100%. B) 300 caracteres generados: texto identico, diferencia 0.0. C) bpc n_eval=10k: motor 1.923309 vs nucleo 1.923309.
## Benchmarks (1 hilo, maquina compartida, ruido)
| historial (chars) | tok/s (bucle numba) | us/token | tok/s bucle Python | p50 us | p99 us | memoria de sesion |
|---|---|---|---|---|---|---|
| 0 | 4,974 | 201 | 6,199 | 112 | 1,063 | 0.09 MB |
| 1,000 | 5,522 | 181 | 5,961 | 128 | 531 | 0.19 MB |
| 10,000 | 4,403 | 227 | 6,288 | 146 | 347 | 3.0 MB |
| 100,000 | 4,884 | 205 | 5,922 | 156 | 368 | 24.1 MB |
Concurrencia (procesos; historial 1k): 1 -> 4,365 tok/s; 2 -> 12,219; 4 -> 22,394; 8 -> 45,360 agregados (16 hilos logicos compartidos). Scoring por lotes: 6,788 pos/s (1 hilo). Nucleo `predict` de 1 posicion crece con el historial (5-97 ms segun L y corrida); el motor queda plano (~0.2 ms/token).
## Economia (supuestos explicitos, NO cotizaciones)
Medido: ~4,946 chars/s por vCPU (913 palabras/s a 5.42 chars/palabra). Costo por 1M chars = P(USD/vCPU-h) x 1e6 / (4946 x 3600) / u. Con P = 0.02/0.05/0.10: 0.0011/0.0028/0.0056 USD por 1M chars; por 1M palabras 0.0061/0.0152/0.0304; por 1M "tokens de 4 chars" 0.0045/0.0112/0.0225. Un token aqui es un caracter: no es comparable con precios de LLM; el limite comercial es la CALIDAD (1.92 bpc en Shakespeare), no el costo.
## Limitaciones / no medido
Un corpus, una semilla, tiempos contaminados; prueba de 8 procesos uso 8 hilos; RSS real no medido (ctypes fallo); servidor HTTPServer monohilo sin autenticacion en 127.0.0.1; memoria por sesion sobredimensionada (~250 B/char); cuello: ~36 busquedas de nivel por posicion (~120 us de ~150 us), cache ~18 us. Sin constantes libres nuevas (capacidad hash 2x es de ingenieria).
## Uso
`IxchelEngine.load(r"...\r3_engine\model_v03")`; `new_session(prompt)`; `generate_fast(sess, n, seed)`. Regenerar: train_save.py y add_hash.py.

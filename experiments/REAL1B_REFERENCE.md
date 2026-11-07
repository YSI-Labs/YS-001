# REAL1B_REFERENCE: costo medido de una red de pesos reales y vara de calibracion (2026-11-06)

Todo lo "medido" sale de `ixchel/results/cost_*_t8.json`, `ref_qwen15.json`, `ref_tinyllama.json` y `big2_eval_v05.log`. Lo marcado **[SUPUESTO]** no se midio aqui.

## 1. Costo medido en ESTA CPU (8 hilos, fp32, decoder tipo Llama con pesos aleatorios; el costo no depende de los valores)
| red | params totales | sin embeddings | entrenar tok/s | prefill tok/s | decodificar tok/s | RAM pico | FLOP/s efectivos (6*N*T) |
|---|---|---|---|---|---|---|---|
| 39M | 38.7M | 14.2M | 895 | 3,923 | 59.0 | 1.4 GB | 76 G |
| 134M | 134.1M | 85.0M | 293 | 1,275 | 31.1 | 2.8 GB | 149 G |
| 374M | 373.9M | 308.3M | 97.8 | 423 | 12.0 | 6.7 GB | 181 G |
| 1.1B | 1,100.0M | 969.0M | **33.2** | 146 | **5.19** | 19.0 GB | 193 G |
El rendimiento satura en ~190 GFLOP/s; el entrenamiento incluye el paso de AdamW (solo para medir tiempo).

## 2. Extrapolacion a esta CPU (formula 6*N*T con N sin embeddings, 193 GFLOP/s medidos)
| objetivo | FLOPs | tiempo en esta CPU |
|---|---|---|
| 1B, 20B tokens | 1.16e20 | ~19 anos |
| 7B (N~6.5B), 1T tokens | 3.9e22 | ~6,400 anos |
Conclusion medida: ninguna red de pesos reales a esta escala se entrena en CPU de escritorio. El pivote "pesos reales" exige GPUs.

## 3. Costo estimado en GPU **[SUPUESTO, no medido]**
Supuestos: una GPU tipo H100 sostiene ~3.5e14 FLOP/s efectivos en bf16 (~35 % de utilizacion de su pico teorico) y se alquila a ~2 USD/hora. Ambos numeros deben verificarse con una cotizacion y un entrenamiento piloto antes de usarse.
| objetivo | GPU-horas | costo (a 2 USD/h) |
|---|---|---|
| 1B, 20B tokens | ~92 | ~185 USD |
| 1B, 100B tokens | ~460 | ~920 USD |
| 7B, 1T tokens | ~31,000 | ~62,000 USD |
Excluye datos, evaluacion, reintentos, fallos y el ingeniero. Una corrida real cuesta tipicamente varias veces la ideal.

## 4. Vara de calibracion (modelos abiertos, misma bateria que Ixchel)
| | Qwen2.5-1.5B | TinyLlama-1.1B | Ixchel v0.5 big2 |
|---|---|---|---|
| bpc web / code / math / chat / books | .925 / .692 / .659 / .696 / 1.294 | .905 / .798 / .815 / .757 / 1.380 | 1.595 / 1.309 / 1.413 / 1.429 / 2.053 |
| HellaSwag acc_norm | 67.2 | 58.4 | 28.8 |
| PIQA acc_norm | 77.5 | 73.0 | 56.5 |
| ARC-Easy acc_norm | 70.5 | 57.3 | 30.1 |
| HumanEval pass@1 | 37.2 % | 0.6 % | 0 % |
| decodificar | 2.2 tok/s (gen. HumanEval) | 4.2 tok/s | ~9.2k pos/s (caracteres) |
Limites: bpc de las referencias sobre 24 segmentos x 1500 chars (no las mismas 30k posiciones); n=400-500 en las pruebas de opcion multiple (SE ~2 puntos); los modelos abiertos pudieron ver el texto de evaluacion durante su preentrenamiento (no verificado). TinyLlama-1.1B base saca 0.6 % en HumanEval: la brecha en programacion exige datos y escala especificos, no solo parametros.

## 5. Lectura
- Un modelo de 1.1B de pesos reales ya supera con mucho a Ixchel en todas las pruebas de capacidad, pero solo se entrena con GPUs.
- Ixchel decodifica unos 3 ordenes de magnitud mas rapido por nucleo (~5k caracteres/s/hilo medidos por el motor incremental frente a 5.2 tokens/s con 8 hilos), pero es un modelo mucho mas debil: la velocidad no compensa la falta de generalizacion.

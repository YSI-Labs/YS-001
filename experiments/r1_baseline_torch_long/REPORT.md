# REPORT r1_baseline_torch_long (orquestador; cifras de result_gru.json y log_gru_1800.txt)
Referencia neuronal con presupuesto largo. GRU 2 capas, d=256, 838,752 params, ctx 128, AdamW lr 3e-3 + coseno (mismo codigo que r1_baseline_torch), presupuesto 1800 s de pared, torch/numba con 8 hilos. CORRIO CONCURRENTEMENTE con los 5 agentes de la ronda 2 (CPU contendida): mide un limite superior del coste.
- val bpc (n_eval=10000): **2.0702**; leak_free=True.
- 2578 pasos, 21.1M tokens vistos (4.6 epocas), fit pared 1810.9 s, fit CPU 11,381 s, 11,731 ch/s de entrenamiento, 664 ch/s de inferencia (sin cache).
- Curva dev (UNA corrida, coseno a 1800 s): 2 min 2.409; 4 min 2.253; 8 min 2.100 (la corrida de 8 min con 2 hilos y presupuesto propio dio 2.314 val). La mejora se aplana hacia el final; no se midio convergencia (sin barrido de tamano ni de tiempo mayor).
- Muestra: texto de forma shakespeareana con palabras mayormente reales y errores de palabra/sintaxis ("KING.\nAt thy second dreadful first live. Are I am quarrel...").
Comparacion honesta con Ixchel v0.1 (2.078 bpc; ~52 s de pared, ~60 s de CPU): misma calidad (+-0.01) con ~190x menos segundos de CPU. Salvedades: n_eval distinto (10k vs 20k), un solo run/semilla, GRU contendida, contexto de la GRU 128 vs Ixchel con orden<=8 (sin cache).

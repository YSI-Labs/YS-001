# REPORT r1_baseline_torch (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de result_transformer.json / result_gru.json / log_*.txt)

Rol: REFERENCIA neuronal estandar (AdamW + backprop permitidos). NO es candidata. Presupuesto fijo: 8 min de pared por modelo, 2 hilos (la GRU uso mas: fit_cpu 993 s > wall 503 s), CPU compartida con ~7 agentes.

| variante | bpc val (n_eval=10k) | bpc dev 8min | params | pasos | chars vistos | train s | ch/s entren. | ch/s inferencia |
|---|---|---|---|---|---|---|---|---|
| transformer d128 L4 h4 ctx128 | 3.593 | 3.620 | 822,016 | 225 | 1.84M (0.40 ep) | 481 | 3,829 | 185 |
| GRU d256 L2 ctx128 | 2.314 | 2.309 | 838,752 | 589 | 4.83M (1.05 ep) | 480 | 10,051 | 349 |
Ref (no reejecutada): WB-5 2.269 bpc, 0.6 s de fit. Mejor sin-gradiente de la ronda 1: Tz'ib KN-5 2.185.
Curva dev (UNA corrida, coseno anclado a 8 min): transformer 2/4/8 min: 3.766/3.698/3.620; GRU: 2.740/2.433/2.309 (la GRU sigue bajando).
Muestra GRU: "Hath h|im I will, "hatch, call'd you? BOTH. When fly scroke in that I do you. It issue; gold. But, Would then--divarian three here come of me; it now it off incent the husband..."
Muestra transformer: ruido de letras ("Hath hil Anthinth t s than asoraysire'ckaldit...").

Hiperparametros usados: AdamW lr 3e-3, betas (0.9,0.95), wd 0.1, batch 64 x ctx 128, clip 1.0, warmup 20, coseno por tiempo de pared, init N(0,0.02) (transformer), semilla 0, dev 5% final de train.

Fallas/salvedades: smoke test dio 39 bpc (embedding N(0,1) atado a la cabeza; arreglado con init nanoGPT); bug de `extra` vacio en harness.run arreglado; transformer muy subentrenado (225 pasos, 0.4 epocas, ~2.1 s/paso): prueba de que 8 min de CPU no alcanzan, no de que sea mal modelo; el corte de 2 min esta sin anneal; GRU min_prob 3.6e-12 (toca el clamp); ch/s no estrictamente comparables (maquina compartida); inferencia sin cache (10k evaluaciones = 29-54 s); escalabilidad no medida.

Lectura para el preprint: en 8 min de CPU, un n-gram que se entrena en <1 s iguala/supera a una GRU de 0.84M params y supera con holgura a un transformer pequeno. Estimacion del agente (juicio, NO medida): ~85% de que una neuronal pequena supere a WB-5 con 2-3x mas computo. Siguiente: GRU a 30 min con el mismo harness y segundo tamano.

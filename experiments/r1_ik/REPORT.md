# REPORT r1_ik (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/*.json, result.json, final.log, sweep.log)

**Veredicto: el espectral PURO es malo (3.2-3.8 bpc); mezclado con WB-5 queda levemente mejor que WB-5 (2.21 vs 2.27), pero la ganancia es casi toda de la mezcla, no del estado espectral.**

## Idea e implementacion
Estado de historia u_t = U_r^T h_t, con U de SVD aleatorizada (Halko, 2 iteraciones) de C = E[h f^T]. h = bolsa hasheada de sufijos de 1..12 chars; f = one-hot del siguiente char + bigrama futuro. Lectura: ridge cerrado sobre [u,1] hacia one-hot(siguiente), clip>=0+eps y normalizacion. Mix opcional p = a*p_spec + (1-a)*p_WB5 con a y lambda elegidos en dev (ultimo 5% del train, nunca val); modelo final entrenado con 95% del train. Fuera: estado recurrente A_c u.

## Resultados (bpc val, n_eval 20k salvo final; params; fit s; train ch/s; infer ch/s)
- pure r16 D2^14: 3.775; 0.26M; 25.8; 187k; 161k
- pure r64 D2^14: 3.446; 1.05M; 33.5; 144k; 95k
- pure r256 D2^14: 3.365; 4.22M; 57.8; 83k; 29k
- pure r1024 D2^14: 3.237; 16.9M; 208.6; 23k; 24k
- pure r256 D2^12: 3.548; 1.07M; 33.3; 145k; 54k
- pure r256 D2^16: 3.200; 16.8M; 57.5; 84k; 70k
- mix r16 D2^14: 2.224; 1.32M; 25.6; 189k; 9.9k
- mix r64 D2^14: 2.210; 2.11M; 39.5; 122k; 8.3k
- mix r256 D2^14: 2.211; 5.27M; 62.4; 77k; 7.4k
- mix r1024 D2^14: 2.210; 17.9M; 164.5; 29k; 6.7k
- mix r256 D2^12: 2.221; 2.13M; 32.1; 150k; 10.7k
- mix r256 D2^16: 2.206 (20k) / 2.213 (50k, final); 17.9M; 56.1; 86k; 11.3k
- CONTROL proyeccion aleatoria (en vez de SVD), mix r256 D2^14: 2.251; pure: 4.311
Ref: WB-5 harness 2.269 (1.08M). Dev: WB-5 (95% train) = 2.221; mix elegido = 2.125. Tiempos contaminados (2 hilos).
Muestra (mejor): "...DUKE.\nHath he.\n\nROSENEN\nSCENE II bid mylock'd, and for her of thing it all sends and whentreat sinew rus uttermost; and,\nThat ths thatKINGHAM..." (forma shakespeareana con errores de palabra; sin coherencia).

## Hiperparametros
~12 constantes. Variables en barrido: r (16..1024), D (2^12..2^16). Elegidas en dev: lambda, alpha. Fijas: L=12, semilla 0, hash, pesos 1 por n-grama, futuro 1 char + bigrama, dev_frac 0.05, n_reg 500k, eps 1e-6, orden WB=5. Reales que cambian el resultado: 2 (r, D) + 2 en dev. Advertencia: en mix el lambda optimo cae en el borde superior de la rejilla (1.0); no se amplio.

## Fallas / hallazgos negativos
- Masa negativa estructural: la lectura lineal da masa negativa en casi todas las filas (pure r256: 12.1% de masa negativa por fila, 100% de filas afectadas; r1024: 4.8%; r16: 3.0%). En mix r256 D2^16: 0.5%, absorbida por WB.
- Pure es malo: ridge lineal sobre 1024 dims no captura la condicional del siguiente char.
- La SVD SI aporta senal frente a proyeccion aleatoria (pure 3.365 vs 4.311) pero en mix la ventaja es solo 0.04 bpc (2.211 vs 2.251).
- La mejora sobre WB es mayormente de la mezcla (suavizado). Falta un control WB-95 evaluado en val para afirmar la ganancia (WB-5 de referencia usa 100% del train y 20k eval): comparacion no exacta.
- Escalar r no ayuda en mix (64->1024 igual); en pure si (3.45->3.24, 16x memoria). Escalar D ayuda poco en mix (~0.015).
- Inferencia mix ~10k ch/s: domina el WB dentro de predict.
- Bug corregido: overflow uint64 escalar en el hash (solo warning).

## Escalabilidad
Duplicar r (pure): 3.45->3.36->3.24; params x4 por paso; fit 33->58->209 s. Duplicar D (pure): 3.55->3.36->3.20. Memoria ~ D*r*4 B (D=2^16, r=256: 17.9M params, ~70 MB).

## Veredicto del agente
~10%. Espectral puro ~1 bpc detras de WB, masa negativa estructural, ganancia de la mezcla pequena y mayormente suavizado WB. Siguiente: lectura no lineal cerrada (features aleatorias/Nystrom sobre u); futuros de 3-4 chars y estado recurrente; control WB-95 en val y rejilla de lambda ampliada; si no baja de ~2.2 con el espectral como componente, abandonar.

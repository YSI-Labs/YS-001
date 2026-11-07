# REPORT r2_scale (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/*.json, powerfit.json, compact_test.json, prof_*.json)

Maquina compartida (varios procesos de otros agentes): tiempos contaminados, mas con 4 hilos. n_eval=20000, leak_free=True en todos.

## 1. Curva de escala por datos
| frac | chars | A bpc | A pares | A fit s | A RAM MB | AWSL bpc | AWSL pares | AWSL fit s | AWSL RAM MB |
|---|---|---|---|---|---|---|---|---|---|
| 6% | 0.29M | 2.5050 | 0.82M | 6.8 | 257 | 2.4681 | 1.76M | 3.1 | 262 |
| 12.5% | 0.60M | 2.3851 | 1.42M | 2.7 | 307 | 2.3355 | 3.12M | 6.6 | 394 |
| 25% | 1.21M | 2.2981 | 2.45M | 5.3 | 473 | 2.2517 | 5.46M | 14.8 | 641 |
| 50% | 2.41M | 2.2240 | 4.07M | 10.8 | 783 | 2.1749 | 9.27M | 33.4 | 1106 |
| 100% | 4.82M | 2.1261 | 6.69M | 22.2 | 1366 | 2.0784 | 15.65M | 74.3 | 1984 |
AWSL 100% = 2.0784 (reproduce v0.1). Cada duplicacion de datos: ~-0.07 a -0.10 bpc (A), sin meseta visible.
Ajuste bpc = a*N^-b + c (5 puntos): A: b=0.111 (se 0.093), c=1.12 (se 0.99); AWSL: b=0.161 (se 0.108), c=1.43 (se 0.56). Extrapolacion por 10x/100x/1000x datos: A 1.90/1.73/1.59; AWSL 1.88/1.74/1.64. **ADVERTENCIA: piso c no acotado (se > valor), 5 puntos cubren 1.5 octavas: cifras orientativas, no prediccion.** Sin piso, con los 3 puntos mayores, b=0.056-0.058 (la pendiente cae en el tramo alto; 10x daria ~1.8-1.9). Pares ~ N^0.75-0.78.

## 2. Throughput (AWSL, train completo; calentamiento JIT excluido)
Recuento: 1/2/4 hilos: core 45.8/45.5/35.9 s -> core_opt 20.6/17.3/16.6 s (2.2x/2.6x/2.2x; 105k->235k, 106k->280k, 135k->291k ch/s). Fit completo (dev+EM+recuento): 95.4->43.5 s (1h), 95.0->36.6 (2h), 77.2->30.7 (4h).
Inferencia, lote de 4096 posiciones (mediana de 5): 1h 241->117 ms (17.0k->35.0k pos/s); 2h 145->63 ms (28.2k->65.3k); 4h 110->40 ms (37.2k->103k pos/s).
Generacion (1 posicion): core_opt 1 hilo ~4.8k ch/s; mas hilos empeoran (overhead de prange).
Cuello de botella (core, 1 hilo): busquedas binarias (~53% del predict), atomos del prefijo completo (~27%); en recuento, np.unique (dos ordenaciones por nivel) = 91%.

## Optimizaciones EXACTAS (experiments/r2_scale/core_opt.py)
OPT1 ordenacion radix estable en numba (1 sort por nivel en vez de 2 np.unique); OPT2 indice hash abierto por nivel en vez de busqueda binaria; OPT3 atomos solo en las posiciones pedidas. Verificadas: probabilidades identicas bit a bit en 1/2/4 hilos (max diff 0.0), tablas y bpc identicos (2.272333 en 1M chars). Coste: RSS 2008 -> 2151 MB (+7%).

## 3. Memoria y latencia
- Actual: 242.6 MB / 15.65M pares = 15.5 B/par (psym 1 + pcnt 4 + ckey 8 + cstart 8 por contexto; 0.656 contextos/par).
- Compacta propuesta (estimada, NO construida): ~7.25 B/par.
- Cuantizacion log de conteos (AWSL 100%, mismos descuentos): exacto 2.07836; 8 bits 2.07839; 7 bits 2.07873 (+0.0003); 5 bits 2.08082 (+0.0025). Colisiones de huella fp32: 0.046% de contextos (bpc con fp32 real NO medido).
- Latencia de busqueda (tabla sintetica, 1 hilo): 10M contextos: binaria 582 ns / hash 111 ns; 100M: 889 / 172 ns (RSS 3085 MB). Indice hash a 100M: 1.07 GB.
- Proyeccion RAM: 1B pares = 15.5 GB (actual) / 7.3 GB (compacta); 10B = 155 / 73 GB; 100B = 1.55 TB / 0.73 TB (extrapolacion lineal).

## 4. Estimacion honesta para 1B de pares
3.24 pares/char medidos; con N^0.78, 1B pares requieren ~1e9 chars (~1 GB de texto, ~200x Shakespeare). Conteo ~1 h a 290k ch/s (4 hilos) sin contar mayor coste de ordenacion; viable en 31 GB solo con formato compacto y particionado. Lo que Shakespeare SI permite afirmar: bpc baja monotonamente con mas datos en 0.3M-4.8M chars sin meseta visible; el modelo es exacto y las optimizaciones no cambian probabilidades. Lo que NO: el valor del piso, si 1B pares superan a redes pequenas, ni RAM/tiempo/latencia reales a 1B (solo extrapolaciones).

## No medido
bpc con huella fp32 real; latencia de inferencia con 1B pares reales; throughput de conteo con mas de 4.8M chars; tokens/s a escala con tablas grandes reales (solo sintetico 100M); 4 hilos en la ruta de inferencia no mejora (posible limite de memoria/contencion, sin explicar).

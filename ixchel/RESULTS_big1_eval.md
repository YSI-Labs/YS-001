# Ixchel grande (914M pares, 656M caracteres) vs Ixchel mediano (200M pares, 81M caracteres) - mismo v0.4, misma bateria

Fuente: ixchel/results/big_v04big.json y mixfinal_v04full.json. Entrenamiento: grande = tablas construidas fuera de memoria (38 min pared, ver RESULTS_big1_build.md) + ajuste de pesos PoE en dev (68 s); mediano = 21 min en memoria.

## bpc por dominio (30,000 posiciones/dominio)
| dominio | mediano (200M pares) | grande (914M pares) | cambio | participacion en el corpus (mix1 -> big1) |
|---|---|---|---|---|
| web | 1.693 | 1.593 | -0.100 | 52.6% -> 69.5% |
| code | 1.405 | 1.355 | -0.050 | 23.4% -> 27.5% |
| math | 1.415 | 1.468 | +0.053 | 4.7% -> 0.6% |
| chat **(CONTAMINADO, invalido)** | 1.522 | 1.205 | -0.317 | 13.7% -> 1.7% |
| books | 1.995 | 2.062 | +0.068 | 5.6% -> 0.7% |

**CONTAMINACION del dominio chat (grande):** `prep_big.py` baraja los documentos de Dolly con un estado aleatorio distinto al de `prep_mix.py`; los 750 documentos de validacion de mix1 quedaron DENTRO del entrenamiento del corpus grande. La caida de 1.52 a 1.21 bpc (top-1 75%) es memorizacion, no mejora real. El resto de dominios no tiene este problema (web: shard 0 vs shards 1-3; codigo: paquetes retenidos; matematicas: train/test oficial; libros: ultimo 10%), aunque C4 puede contener duplicados y los benchmarks podrian aparecer en la web (no medido). Se corregira en la proxima reconstruccion excluyendo por contenido los documentos de val_chat.

## Opcion multiple (acc_norm; azar 25/50/25)
| prueba | mediano | grande |
|---|---|---|
| hellaswag | 28.2% +-0.6 | 28.9% +-0.6 |
| piqa | 54.7% +-1.2 | 56.3% +-1.2 |
| arc_easy | 29.4% +-0.9 | 30.4% +-0.9 |

HumanEval pass@1: grande 0/164; completaciones no triviales 17.7%, validas entre ellas 0.6%.
Validez sintactica de continuaciones sobre codigo no visto: grande 16.7% (mediano 15.8%; continuacion real 27.5%).

## Costo de ejecucion del modelo grande (tablas mmap, 13.6 GB; maquina libre; 4 hilos numba)
- predict por lotes de 4096 posiciones: 16,458 posiciones/s; predict de 1 posicion (historial 3000): 0.60 ms.
- carga de tablas: 0.4 s (mmap); ajuste de pesos PoE sobre 150,000 posiciones dev: 68 s; PoE dev 1.456 bpc.

## Lectura honesta
- 4.6x mas pares y 8x mas datos dan mejoras reales pero pequenas: web -0.10, codigo -0.05 bpc; las pruebas de opcion multiple suben 0-1.6 puntos (dentro de ~2 SE) y siguen a 3-6 puntos del azar; HumanEval 0/164.
- DILUCION DE DOMINIO: al pasar matematicas de 4.7% a 0.6% y libros de 5.6% a 0.7% del corpus, su bpc EMPEORA (+0.05 y +0.07) pese a tener mas datos totales: con conteo exacto, la mezcla de datos es una palanca de primer orden (equivalente a la mezcla de datos de un LLM).

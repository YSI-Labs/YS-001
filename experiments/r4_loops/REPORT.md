# REPORT r4_loops (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/gen_*.json, qwen_*.json, bpc_*.json)

**Veredicto: ninguna receta con cero constantes mejora la generacion de forma util; la receta (d) empeora el bpc sobre texto real. No se cambia la base.**
Lectura clave: la repeticion y la perplejidad bajo un modelo real son senales DISTINTAS. Qwen2.5-1.5B premia la repeticion (texto con bucles a T=0.8 obtiene bpc bajo), asi que la perplejidad sola no mide calidad: hay que leer ambas juntas.

## Base y muestreo (90 textos de 400 caracteres por variante; Qwen sobre 30)
| configuracion | rep4 | maxrep | Qwen bpc |
|---|---|---|---|
| base T=1.0 | 0.004 | 45 | 3.21 |
| base T=0.8 | 0.111 | 63.5 | 1.69 |
| texto REAL del mismo dominio | 0.045 | 18 | 0.83 |
| top-p 0.9 a T=1.0 (hiperparametro, solo comparacion) | 0.069 | 80 | 1.81 |
| penalizacion 1.3 a T=1.0 (hiperparametro, solo comparacion) | 0.030 | 49 | 3.08 (n.s.) |
Los bucles aparecen solo con T<1. Con T=1.0 la base casi no repite pero produce texto ininteligible (3.2 bits por caracter bajo Qwen frente a 0.83 del texto real).

## Recetas (T=1.0)
| receta | rep4 | maxrep | Qwen bpc | vs base (SE) |
|---|---|---|---|---|
| a: cache solo sobre el prompt | 0.001 | 11.3 | 3.472 | +0.26 (0.14) |
| b: prompt + 256 | 0.005 | 41.6 | 3.213 | +0.00 (0.13) |
| c1: cache x P/p | 0.000 | 9.9 | 4.726 | +1.51 (0.16) |
| c2: cache x (1-rep4) | 0.004 | 45.5 | 3.224 | +0.01 (0.02) |
| d: retirar si argmax = c1 | 0.001 | 10.3 | 3.641 | +0.43 (0.14) |
| d a T=0.8 | 0.103 | 38.6 | 1.965 | (base T=0.8: 1.687) |
a y c1 eliminan el texto repetido pero el resultado es ruido; c2 y b no cambian nada; d a T=0.8 sigue repitiendo.

## bpc sobre texto real (mix15 val, 30k posiciones/dominio; delta vs base, SE por bloques)
Receta d: web +0.0075 (SE .0011), code +0.164 (SE .016), math +0.034, chat +0.016, books +0.031. a, b, c1, c2 son identidad sobre texto real por construccion (sin texto generado); su kernel reproduce `predict` con diferencia 0.0.

## Limitaciones
30 textos por variante bajo Qwen (SE 0.14-0.26 bpc; c2 y b no detectables); metricas de repeticion con 90 textos de un solo tipo de prompt; HellaSwag/PIQA/ARC no medidos para (d); Qwen en fp32 (~6 GB) por encima del limite de 4 GB del brief (proceso aparte).

## Lectura del orquestador
El problema de calidad de generacion no es solo el cache: el modelo de conteo de caracteres, muestreado a T=1.0 (su distribucion real), produce texto incoherente (Qwen 3.2 vs 0.83 bpc); a T=0.8 la calidad "mejora" segun Qwen solo porque aparecen bucles. Arreglar los bucles sin una fuente de coherencia a mayor escala (representaciones aprendidas) no produce texto util. Esto es coherente con HumanEval 0/164 y las pruebas de sentido comun.

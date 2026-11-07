# BRIEF RONDA 4 - variaciones rapidas sobre Ixchel v0.4 (leer entero)

## Contexto (todo medido; ver ixchel/RESULTS_*.md)
Ixchel v0.4 = cadenas KN (A,W,S,L,Kb,Kc,WL,W2) + cache de historial orden 8 (KeyedWin C8, 4 ventanas) + mezcla geometrica global por Newton (poe_global). Sobre
`mix1` (85.6M chars) y `big1` (656M chars, 914M pares): bpc web ~1.6-1.7, codigo ~1.4, matematicas ~1.4-1.5, libros ~2.0. Un transformer real (Qwen2.5-1.5B) da 0.93/0.69/0.66/1.29 (web/codigo/mat/libros) sobre el mismo texto.
HumanEval 0/164; HellaSwag/PIQA/ARC-E 3-6 puntos sobre el azar. Defectos conocidos: (a) la generacion cae en BUCLES de repeticion (el cache copia el propio texto generado); (b) dilucion de dominio (matematicas/libros empeoran cuando su fraccion del corpus baja); (c) la compuerta por EM aportaba poco y el PoE usa UN solo vector de pesos global.
Objetivo de la ronda: mejoras medibles para un producto (mejor bpc por dominio y generacion que no degenere), SIN learning rates ni optimizadores con hiperparametros libres.

## Entorno de experimento RAPIDO
- Corpus de desarrollo `data/mix15` (15M chars de entrenamiento con las proporciones de mix1; val_* identicos a mix1). `IXCHEL_DATA=mix15`.
- Lee `ixchel/run_mix15.py` (plantilla: `base_model()`, `evaluate()`), `ixchel/core_v04.py`, `ixchel/longcache.py`, `ixchel/mixing.py`, `ixchel/eval_suite.py`. LINEA BASE ya medida en `ixchel/results/mix15_base.json` y `mix15_base_nll.npz` (nll por posicion y dominio, para tests PAREADOS con `eval_suite.domain_bpc`: mismas posiciones). Si aun no existe, espera unos minutos (se esta generando).
- COPIA lo que necesites a `experiments/r4_<codename>/` (no modifiques `ixchel/` ni `common/`). Hilos: `NUMBA_NUM_THREADS=2`, torch 2 hilos. RAM <= 4 GB por proceso. Cada ajuste ~5-8 min con 2 hilos (maquina compartida; hay una evaluacion de un modelo de 1.5B corriendo). No uses val para elegir: dev interno (5% final del train); el reporte final usa los val_*.
- Reporta SIEMPRE: bpc por dominio (web, code, math, chat, books) con diferencia PAREADA contra la base y su error estandar por bloques de 500 posiciones; opcion multiple solo si el cambio podria afectarla.
- Honestidad: cifras solo de tus JSON; reporta lo que falle; audita constantes libres nuevas (prefiero cero).
- No puedes escribir REPORT.md: entrega el informe COMPLETO (<=70 lineas, espanol) como mensaje final con codigo/parche fusionable (ruta) y la tabla de resultados.

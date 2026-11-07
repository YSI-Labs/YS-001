# Estructura del repositorio

## Raíz

| Archivo | Para qué |
|---|---|
| `README.md` | Presentación, resultados clave, requisitos y uso rápido |
| `LICENSE` | Licencia MIT del código |
| `LICENSE-CC-BY-4.0.md` | Licencia CC BY 4.0 del paper y la documentación |
| `.gitignore` | Excluye datos, pesos, cachés y artefactos de Python |

## `preprint/`

El paper de la versión 1.0.

- `PREPRINT.md`: texto fuente (Markdown). Es la versión de referencia.
- `Ixchel_preprint.pdf`: PDF generado a partir de `PREPRINT.md`.
- `build_pdf.py`: genera el PDF con reportlab.
- `figures/`: cuatro figuras y `make_figures.py`, que las genera a partir de los datos de las tablas del paper.

## `ixchel/`

El núcleo de Ixchel y sus herramientas.

- **Versiones del modelo:** `core.py` (v0.1), `core_v02.py` (caché y mezcla geométrica), `core_v04.py` (caché de orden 8), `core_v05.py` y `chains_v05.py` (átomos de código podados).
- **Mezcla y caché:** `mixing.py` (Newton para los pesos), `cache_experts.py`, `longcache.py` (ventanas de historial).
- **Construcción de tablas:** `bigbuild.py` (fuera de memoria), `bigbuild_v1_ram.py` (versión en RAM, para comparar).
- **Ejecuciones:** `run_v01.py` a `run_v04.py`, `run_mix15.py`, `run_mix_full.py`, `run_bigbuild2.py`, `run_big2_eval.py`, `run_pipeline_big2.sh`.
- **Evaluación:** `eval_suite.py` (bpc por dominio, opción múltiple y HumanEval con aislamiento), `eval_ref_model.py` (referencias abiertas), `measure_train_cost.py` (coste de redes).
- **Aplicaciones (§13):** `niche1_*.py` (calidad de datos), `filter_*.py` (entrenar con y sin filtro), `niche4_*.py` (compresión).
- **Resultados:** `results/` (JSON, logs y tablas) y `RESULTS_*.md` (informes por etapa).

## `common/`

- `harness.py`: carga de datos, evaluación pareada y generación de muestras.
- `prep.py`, `prep_text8.py`, `prep_mix.py`, `prep_big.py`, `prep_big2.py`: construcción de los corpus a partir de `data/raw*/`.
- `baseline_wb.py`: referencia de n-grama Witten-Bell.

## `experiments/`

Exploración de las rondas 1 a 4. Cada carpeta `rN_<nombre>/` contiene el código de una arquitectura o variante, sus resultados y su `REPORT.md`.

- **Ronda 1:** `r1_tzolkin`, `r1_pixan`, `r1_ik`, `r1_tzib`, `r1_chan`, `r1_ixim`, `r1_yaxche`, `r1_baseline_torch*`.
- **Ronda 2:** `r2_cache`, `r2_mixer`, `r2_lexstruct`, `r2_classes`, `r2_scale`.
- **Ronda 3:** `r3_lrfree`, `r3_residual`, `r3_longcache`, `r3_engine`, `r3_gru_paired`, `r3_gru_mix`.
- **Ronda 4:** `r4_gate`, `r4_loops`, `r4_codeatoms`.
- `BRIEF*.md` y `ROUND1_SUMMARY.md`: instrucciones internas que se dieron a los agentes y resumen de la ronda 1.

Los informes `REPORT.md` fueron transcritos por el autor a partir de lo que devolvieron los agentes; las cifras se contrastaron con los JSON de cada carpeta.

## `gpu_pilot/`

Piloto para la etapa en GPU: `prep_data.py` (tokenizador de byte propio), `train.py` (AdamW frente a DoG/L-DoG), `dog.py` (optimizador sin tasa de aprendizaje), `compare_runs.py` (tabla y puerta de decisión) y su `README.md`. Probado en CPU; no ejecutado en GPU.

## `docs/`

Esta carpeta: estructura, reproducibilidad, datos y licencias, y notas de publicación.

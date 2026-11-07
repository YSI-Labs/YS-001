# Datos, origen y licencias

Este repositorio **no redistribuye** ningún corpus de terceros. Esta página indica qué archivos espera cada script en `data/`, de dónde obtenerlos y bajo qué licencia. **Verifica la licencia en la fuente antes de usar los datos para algo distinto de la investigación académica.**

## Datos de entrada

| Archivo esperado (en `data/`) | Contenido | Usado por | Fuente de referencia |
|---|---|---|---|
| `raw/pg100.txt` | Obras completas de Shakespeare (Gutenberg #100) | `common/prep.py` | Proyecto Gutenberg, libro 100 |
| `raw_text8/text8` | text8 (100 M de caracteres, a–z y espacio) | `common/prep_text8.py` | Matt Mahoney, mattmahoney.net/dc/text8.zip |
| `raw_mix/c4/c4_part.json.gz` | Primeros ~120 MB de C4 (shard 0), JSON por línea con campo `text` | `common/prep_mix.py` | allenai/c4 (Hugging Face) |
| `raw_mix/c4big/c4_0000{1,2,3}.json.gz` | Shards 1 a 3 de C4 | `common/prep_big.py`, `prep_big2.py`, `niche1_quality.py`, `filter_prepare.py` | allenai/c4 (Hugging Face) |
| `raw_mix/gsm8k/train.jsonl`, `test.jsonl` | Problemas de matemáticas (campos `question`, `answer`) | `prep_mix.py`, `prep_big*.py`, `eval_suite.py` | openai/gsm8k |
| `raw_mix/dolly/dolly.jsonl` | Instrucciones y respuestas | `prep_mix.py`, `prep_big*.py` | databricks/databricks-dolly-15k |
| `raw_mix/hellaswag/validation.parquet` | HellaSwag, validación | `eval_suite.py`, `eval_ref_model.py` | Rowan/hellaswag (Hugging Face) |
| `raw_mix/piqa/valid.jsonl`, `valid-labels.lst` | PIQA, validación | `eval_suite.py`, `eval_ref_model.py` | ybisk/piqa (Hugging Face) |
| `raw_mix/arc/easy_test.parquet` | ARC-Easy, prueba | `eval_suite.py`, `eval_ref_model.py` | allenai/ai2_arc (Hugging Face) |
| `raw_mix/humaneval/HumanEval.jsonl.gz` | HumanEval (164 problemas) | `eval_suite.py`, `eval_ref_model.py` | openai/human-eval |
| Código Python local | Archivos `.py` de la biblioteca estándar del intérprete | `prep_mix.py` y `prep_big*.py` (sección de código) | Tu propia instalación de Python |
| `ref_models/qwen15`, `ref_models/tinyllama` | Pesos de Qwen2.5-1.5B y TinyLlama-1.1B | `eval_ref_model.py`, `niche1_qwen.py` | Qwen/Qwen2.5-1.5B, TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T |

Las rutas exactas de los archivos `raw_mix/...` son las que leen los scripts; si descargas los datos con otro formato, ajusta los nombres.

> **Nota sobre las referencias de origen.** Las identificaciones de Hugging Face y de GitHub de la tabla son las fuentes habituales de cada conjunto. No se guardaron en el repositorio los comandos exactos con que se descargaron; verifica que la versión coincide con la que leen los scripts (por ejemplo, el número de shards de C4).

## Licencias de los datos

| Conjunto | Licencia según la fuente | Notas para este proyecto |
|---|---|---|
| Shakespeare (Gutenberg #100) | Dominio público | Sin restricciones. |
| text8 | Dominio público (según su autor) | Verificar en mattmahoney.net antes de redistribuir. |
| C4 (allenai/c4) | ODC-BY | Requiere atribución. Usado solo para entrenar y evaluar; no se redistribuye. |
| GSM8K | MIT | Sin restricciones relevantes. |
| Dolly (databricks-dolly-15k) | CC BY-SA 3.0 | Incluye material de Wikipedia en algunas categorías. Una red entrenada con él puede tener obligaciones de *share-alike*: revisar antes de un uso comercial. |
| HellaSwag | MIT | Sin restricciones relevantes. |
| PIQA | Verificar en su ficha de Hugging Face | — |
| ARC (AI2) | CC BY-SA 4.0 | Verificar en su ficha. |
| HumanEval | MIT | Sin restricciones relevantes. |
| Qwen2.5-1.5B | Apache-2.0 | Solo se usa como referencia de evaluación; no se redistribuye. |
| TinyLlama-1.1B | Apache-2.0 | Solo referencia; no se redistribuye. |

Las licencias marcadas como "verificar" no se confirmaron en las fuentes durante la preparación de este repositorio.

## Qué se publica de los resultados

- **Sí:** tablas, JSON de métricas, informes de los agentes, figuras y el paper.
- **No:** las tablas de conteo, los pesos entrenados y los corpus generados (todo en `data/` y, por tamaño, en cachés).
- **Textos de terceros:** en `ixchel/results/` hay muestras generadas por los modelos (por ejemplo, `gen_*.json` de `r4_loops`). Algunas muestras reproducen fragmentos de C4. Los correos electrónicos que aparecían en esas muestras se han sustituido por `[correo-redactado]` (ver [`NOTAS_DE_PUBLICACION.md`](NOTAS_DE_PUBLICACION.md)).

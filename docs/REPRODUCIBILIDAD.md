# Reproducibilidad

Este documento explica cómo reproducir los resultados del paper, en qué orden ejecutar los scripts, cuánto tardan y qué recursos hacen falta. Para los datos de terceros, ver [`DATOS_Y_LICENCIAS.md`](DATOS_Y_LICENCIAS.md).

Todas las rutas son relativas a la raíz del repositorio. Los comandos se ejecutan desde esa raíz con `python -I` (modo aislado, que no carga módulos del directorio actual).

## 0. Preparación del entorno

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (en Linux/macOS: source .venv/bin/activate)
pip install numpy numba scipy pandas pyarrow matplotlib zstandard brotli pyppmd tokenizers
pip install torch transformers    # solo para referencias neuronales, piloto y niche1_qwen.py
```

Crea la carpeta `data/` en la raíz y coloca allí los archivos descritos en `DATOS_Y_LICENCIAS.md`.

## 1. Experimento mínimo: Shakespeare (≈ minutos)

```bash
python -I common/prep.py                  # crea data/shakespeare/
python -I ixchel/run_v01.py AWSL 20000    # Ixchel v0.1, bpc en validación
python -I ixchel/run_v04.py               # Ixchel v0.4 (caché de historial de orden 8)
```

Prueba pareada contra la GRU de referencia (`experiments/r3_gru_paired/run.py`, ~1 h en CPU) y `ixchel/RESULTS_paired.md`.

## 2. Corpus mixto `mix1` y `mix15` (≈ horas)

```bash
python -I common/prep_text8.py            # solo si se usa text8 (§9.1)
python -I common/prep_mix.py              # crea data/mix1/ (requiere data/raw_mix/)
python -I ixchel/run_mix15.py             # corpus reducido de 15 M para iterar
python -I ixchel/run_mix_full.py          # evaluación completa sobre mix1
```

## 3. Modelo grande `big2` con Ixchel v0.5 (≈ 1.5 h de construcción + evaluación)

```bash
python -I common/prep_big2.py             # crea data/big2/ (excluye chat de validación)
python -I ixchel/run_bigbuild2.py         # construye las tablas fuera de memoria (2 tandas)
python -I ixchel/run_big2_eval.py v05     # ajuste de la mezcla y batería completa
```

`ixchel/run_pipeline_big2.sh` encadena los pasos anteriores. Requiere ≈ 17 GB de disco para las tablas. Si tu equipo tiene menos RAM, `bigbuild.py` escribe en trozos y no necesita tener el corpus completo en memoria.

**Aviso:** el corpus `big1` (`common/prep_big.py`) tiene una contaminación documentada en el chat de validación. Sus resultados de chat son inválidos; usa `big2` (ver §10.2 del paper).

## 4. Referencias abiertas (≈ 2–3 h por modelo, CPU)

```bash
python -I ixchel/eval_ref_model.py <ruta_al_modelo> <etiqueta>
```

Por ejemplo, con Qwen2.5-1.5B o TinyLlama-1.1B descargados de Hugging Face. Usa la misma batería que Ixchel (bpc por dominio, HellaSwag, PIQA, ARC-Easy, HumanEval) y guarda `ixchel/results/ref_<etiqueta>.json`.

## 5. Costes de cómputo de redes reales (≈ 1 h)

```bash
python -I ixchel/measure_train_cost.py 25m
python -I ixchel/measure_train_cost.py 125m
python -I ixchel/measure_train_cost.py 350m
python -I ixchel/measure_train_cost.py 1b  # requiere ~19 GB de RAM
```

## 6. Aplicaciones: calidad de datos y compresión (§13)

Orden recomendado:

```bash
python -I ixchel/niche1_quality.py        # genera 500 documentos de C4 y las puntuaciones
python -I ixchel/niche1_qwen.py           # perplejidad de Qwen2.5-1.5B sobre 200 documentos (~11 min)
python -I ixchel/niche1_analysis.py       # AUC, IC y Spearman
python -I ixchel/niche1_nocache.py        # variante sin caché de historial
python -I ixchel/niche1_nocache_analysis.py

python -I ixchel/filter_prepare.py        # pool de 8000 documentos y puntuaciones
python -I ixchel/filter_train.py N_all 0  # un brazo y una semilla; repetir para cada brazo y semilla
python -I ixchel/filter_analysis.py

python -I ixchel/niche4_compress.py       # cota de código ideal frente a compresores estándar
python -I ixchel/niche4_ppmd.py           # PPMd (órdenes 8, 16, 32)
python -I ixchel/niche4_coder.py 20000    # codificador aritmético con prueba de ida y vuelta
```

Los brazos del experimento de filtrado son `N_all`, `N_rand50`, `N_low50`, `N_high50`, `C_unf`, `C_filtA`, `C_filtFull` y `C_oracle`, con semillas 0 y 1. Cada entrenamiento tarda unos 10–12 min en 2 hilos.

## 7. Piloto en GPU (preparado, no ejecutado)

Ver [`../gpu_pilot/README.md`](../gpu_pilot/README.md). Se verificó en CPU con un modelo diminuto, pero **no** se ejecutó en CUDA.

## 8. Regenerar el PDF del paper

```bash
cd preprint
python -I build_pdf.py                    # usa reportlab; las figuras se regeneran con figures/make_figures.py
```

## Lo que se puede y lo que no se puede verificar

- **Determinismo.** Las tablas y las mezclas son deterministas dado el mismo corpus. Los resultados de redes neuronales dependen de la semilla y del hardware; el paper reporta una semilla (o dos en §13.3).
- **Tiempos.** Los tiempos del paper son de un equipo concreto (AMD Ryzen 7 7730U, 8 núcleos, 31 GB de RAM, Windows 11) y bajo carga de otros procesos. Tu equipo puede tardar más o menos.
- **Compresor.** El codificador de `niche4_coder.py` se verificó dentro de un mismo proceso. No se ha verificado que el descompresor reproduzca las mismas probabilidades entre máquinas distintas.
- **Referencias de text8.** Las cifras publicadas de Transformer-XL, Al-Rfou, mLSTM y HM-LSTM se contrastaron con resúmenes y tablas de los artículos, no con los PDF completos.

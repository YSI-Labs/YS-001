# Piloto de pesos reales en GPU (125M -> 350M): AdamW (control) frente a DoG / L-DoG

Objetivo (decisión de `experiments/GO_TO_MARKET.md` §4, etapa 1): con el mismo presupuesto de datos y cómputo, comprobar si un optimizador sin tasa de aprendizaje (DoG) se acerca a AdamW en un transformer pequeño. Es una prueba de la **regla "sin AdamW" del proyecto**, no de la arquitectura Ixchel: aquí el modelo es un decoder tipo Llama estándar.

## Qué hay
| archivo | para qué |
|---|---|
| `prep_data.py` | entrena un BPE de byte PROPIO (sin licencia ajena) y tokeniza a `train.bin`/`val.bin` (uint16); val se separa por DOCUMENTOS |
| `train.py` | entrenamiento (bf16 en CUDA), `--optim adamw|dog|ldog`, guarda checkpoint HF para evaluar |
| `dog.py` | DoG global y por capa (L-DoG), sin tasa de aprendizaje, sin schedule |
| `compare_runs.py` | tabla de val loss y la puerta de decisión |

## Estado verificado (2026-11-06, en ESTA CPU, modelo diminuto) y lo que NO está verificado
- Verificado: `prep_data.py` → `train.py` con los tres optimizadores (la pérdida baja en los tres), reanudación con `--resume`, checkpoint HF → `ixchel/eval_ref_model.py` (bpc por dominio, HellaSwag/PIQA/ARC-E, HumanEval) sin cambios.
- **No verificado: nada en CUDA.** `bf16 autocast`, `fused AdamW` y `--compile` no se han ejecutado aquí (no hay GPU). Haz primero la prueba de humo de la etapa 0 en la máquina alquilada. Los valores de AdamW por tamaño (6e-4, 3e-4, 2e-4; β=(0.9,0.95); wd 0.1; coseno al 10 %) son práctica habitual **[M]**, no medidos aquí.
- DoG: la referencia es Ivgi et al. 2023 **[M, verificar la cita]**. Nuestra única medición (GRU 0.84M, 2600 pasos) dio DoG +0.099 bpc sobre AdamW; en el humo diminuto (300 pasos) DoG fue más lento que AdamW. No hay evidencia a favor de DoG en transformers; el piloto existe para obtenerla.

## Requisitos de la máquina alquilada
Python 3.10+, `torch` con CUDA, `transformers`, `tokenizers`, `numpy`, `pandas` (+ `pyarrow` si hay parquet). Una GPU de 24 GB o más basta para 125M; 350M con `--micro-bsz 8`.

## Etapas con puertas de decisión
**Etapa 0 – humo en la GPU (< 10 min).** Con cualquier texto pequeño:
```bash
python prep_data.py --out data_smoke --vocab 4096 --val-frac 0.05 --inputs muestra.txt
python train.py --data data_smoke --out run_smoke --size tiny --optim adamw --tokens 3e6 --seq 256 --micro-bsz 16 --batch-tokens 4096 --eval-every 100
```
Puerta: la pérdida baja y no hay NaN. Después repetir con `--compile` y decidir si se usa.

**Etapa 1a – 125M con los datos que ya tienes (barata).** Los shards de C4 locales (`data/raw_mix/c4big/*.json.gz`, ~1 GB comprimido) dan unos 150–200 M de tokens (estimación, depende del BPE): son varias épocas, así que el resultado vale para comparar optimizadores, no como medida de calidad final. Solo web: no usar los bpc de código/matemáticas para comparar con Ixchel.
```bash
python prep_data.py --out data_c4 --vocab 32000 --inputs c4_00001.json.gz c4_00002.json.gz c4_00003.json.gz
for o in adamw dog ldog; do python train.py --data data_c4 --out run_125m_$o --size 125m --optim $o --tokens 5e8 --seed 0; done
python compare_runs.py run_125m_adamw run_125m_dog run_125m_ldog
```
**Puerta de la etapa 1a:** `dog - adamw` en val loss final ≤ **+0.05 nats/token** → se puede mantener la regla "sin AdamW" (pasar a 1b con DoG). Si no pasa → el modelo comercial usa AdamW (decisión tomada con datos, no por principio). Repetir con `--seed 1` para conocer la dispersión: si la diferencia es menor que la dispersión entre semillas, no concluir.

**Etapa 1b – 125M con datos nuevos (2.5 B de tokens = 20 tokens/parámetro).** Hace falta un corpus mayor de licencia verificada (en la máquina alquilada, no incluido aquí): p. ej. FineWeb-Edu, un subconjunto permisivo de código y un conjunto de matemáticas **[M: nombres y licencias a comprobar antes de usar, incluidos los términos de uso para entrenar un modelo comercial]**. Mismos comandos con `--tokens 2.5e9`.

**Etapa 2 – 350M (solo si 1b pasa).** `--size 350m --tokens 7.5e9 --micro-bsz 8`.

## Costes **[SUPUESTO, no medido]**
Supuestos: ~3.5e14 FLOP/s efectivos en bf16 (≈35 % del pico de una H100) y 2 USD por GPU-hora; ambos hay que verificar con una cotización y con la etapa 0. Fórmula 6·N·T con N = parámetros totales.
| corrida | tokens | FLOPs | GPU-horas | coste ideal |
|---|---|---|---|---|
| 125M, etapa 1a (×3 optimizadores) | 5e8 cada una | 4.0e17 | 0.3 h cada una | ~2 USD en total |
| 125M, etapa 1b (×2) | 2.5e9 cada una | 2.0e18 | 1.6 h cada una | ~3 USD cada una |
| 350M (×2) | 7.5e9 cada una | 1.7e19 | 13 h cada una | ~27 USD cada una |
Son costes **ideales**: sin datos, fallos, reintentos, evaluación ni tu tiempo. El rango "100–500 USD" de `GO_TO_MARKET.md` es un tope prudente que cubre todo eso; si los números de aquí se cumplen, el piloto cuesta una fracción.

## Qué traer de vuelta
Por cada corrida: `log.jsonl`, `final.json` y la carpeta `hf/` (≈0.5 GB para 125M). Después, en esta máquina:
```bash
cd ixchel
python -I eval_ref_model.py ../gpu_pilot/run_125m_adamw/hf pilot125_adamw 8
python -I eval_ref_model.py ../gpu_pilot/run_125m_dog/hf pilot125_dog 8
```
(~1 h cada una en CPU; incluye HumanEval.) Los bpc salen en las mismas posiciones y con la misma fórmula que las referencias de Qwen2.5-1.5B y TinyLlama-1.1B (`experiments/REAL1B_REFERENCE.md`).

## Qué NO concluir de este piloto
- Un 125M entrenado en C4 no se acercará a TinyLlama-1.1B en HellaSwag/PIQA/ARC; el piloto solo compara optimizadores y calibra costes reales.
- La puerta de 0.05 nats la fijé yo antes de medir: es una decisión de ingeniería, no un umbral estadístico.
- Un resultado favorable a DoG a 125M no garantiza lo mismo a 1B o 7B.

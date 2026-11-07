# YS-001 · Ixchel: modelado de lenguaje por mezcla de expertos de conteo

Código, documentación y paper del proyecto **Ixchel**, una familia de modelos de lenguaje a nivel de carácter que **no usa atención, retropropagación ni optimizadores con tasa de aprendizaje**. Todo el entrenamiento son conteos exactos, suavizado Kneser-Ney y una mezcla geométrica cuyos pesos se ajustan por máxima verosimilitud con Newton (sin tasa de aprendizaje). Se entrena en CPU de escritorio.

> **Autor:** laboratório de pesquisa e desenvolvimento
> **Paper:** [`preprint/Ixchel_preprint.pdf`](preprint/Ixchel_preprint.pdf) (versión 1.0, noviembre de 2026)
> **Licencia:** código bajo [MIT](LICENSE); paper y documentación bajo [CC BY 4.0](LICENSE-CC-BY-4.0.md)

---

## Resumen

Ixchel obtiene resultados sólidos frente a una red neuronal pequeña entrenada con gradientes, pero **no generaliza** a tareas de razonamiento ni de programación. El repositorio documenta con el mismo detalle los resultados positivos, los fallidos y las limitaciones.

| Experimento | Resultado | Referencia |
|---|---|---|
| Shakespeare, Ixchel v0.3 frente a GRU 0.84 M (AdamW), con el mismo contexto | **−0.057 ± 0.014 bpc** a favor de Ixchel (test pareado) | §9.2 del paper |
| text8 (90 M de caracteres, test estándar) | 1.419 bpc | §9.1 |
| Corpus mixto, modelo grande de 1.17 mil millones de celdas (Ixchel v0.5) | web 1.595 · código 1.309 · matemáticas 1.413 · chat 1.429 · libros 2.053 bpc | §10.2 |
| Modelos abiertos de referencia (web) | TinyLlama-1.1B 0.905 · Qwen2.5-1.5B 0.925 bpc | §10.2 |
| HellaSwag / PIQA / ARC-Easy (Ixchel grande) | 28.8 % / 56.5 % / 30.1 % (azar: 25 / 50 / 25 %) | §10.2 |
| HumanEval (Ixchel grande) | 0 / 164 | §10.2 |
| Entrenar con datos filtrados (GRU 0.84 M, 2 semillas) | filtrar basura mejora 0.019 bpc; un n-grama KN lo consigue igual que Ixchel | §13.3 |
| Compresión frente a PPMd (cota de código ideal) | 8–32 % menor tamaño; el modelo pesa 16.9 GB | §13.2–13.3 |

**Conclusión del paper:** Ixchel es una línea base sin gradientes sólida y reproducible. No es, por sí sola, un modelo de uso general. El camino hacia uno pasa por redes de pesos aprendidos entrenadas en GPU; el repositorio incluye un piloto preparado para esa etapa en `gpu_pilot/`.

---

## Estructura del repositorio

```
YS-001/
├── preprint/        Paper (PREPRINT.md, PDF), figuras y script de generación del PDF
├── ixchel/          Núcleo de Ixchel (v0.1 a v0.5), evaluación, construcción fuera de memoria
├── common/          Harness de evaluación y preparación de corpus
├── experiments/     Rondas 1 a 4 de exploración (un directorio por arquitectura o variante)
├── gpu_pilot/       Piloto para GPU: AdamW frente a DoG/L-DoG (preparado, no ejecutado)
└── docs/            Estructura detallada, reproducibilidad, datos y licencias, notas de publicación
```

Más detalle en [`docs/ESTRUCTURA.md`](docs/ESTRUCTURA.md).

## Requisitos

- Python 3.10 o superior.
- Paquetes: `numpy`, `numba`, `scipy`, `pandas`, `pyarrow`, `matplotlib`, `zstandard`, `brotli`, `pyppmd`, `tokenizers`, `transformers` y `torch` (solo para las referencias neuronales y el piloto).
- Para los corpus grandes: unos **32 GB de RAM** y **≈100 GB de disco** libres. El modelo grande ocupa 16.9 GB en tablas.
- El paper se midió en un equipo con AMD Ryzen 7 7730U (8 núcleos), 31 GB de RAM y Windows 11.

Instrucciones paso a paso en [`docs/REPRODUCIBILIDAD.md`](docs/REPRODUCIBILIDAD.md).

## Inicio rápido

El primer experimento usa solo las obras de Shakespeare, que se obtienen del Proyecto Gutenberg (libro 100):

```bash
python -I common/prep.py                      # prepara data/shakespeare (requiere el texto descargado)
python -I ixchel/run_v01.py AWSL 20000        # Ixchel v0.1 sobre Shakespeare
```

Todos los corpus se colocan en `data/` (no incluido en el repositorio). Ver [`docs/DATOS_Y_LICENCIAS.md`](docs/DATOS_Y_LICENCIAS.md).

## Qué no está incluido

- **Datos de terceros** (C4, Dolly, GSM8K, HellaSwag, PIQA, ARC, HumanEval, text8, Shakespeare): no se redistribuyen. Se indica dónde obtenerlos y bajo qué licencia.
- **Modelos entrenados y tablas de conteo** (`*.npy`, `*.npz`, `*.pt`, `*.safetensors`, `*.bin`): pesan decenas de GB y se regeneran con los scripts.
- **Modelos de referencia** (Qwen2.5-1.5B, TinyLlama-1.1B): se descargan de sus fuentes oficiales.

## Limitaciones

- Los corpus de las rondas 1 y 2 son pequeños (Shakespeare, text8) y una sola semilla por configuración. Las comparaciones con redes neuronales usan una GRU de 0.84 M de parámetros; una red mayor podría cambiar el resultado.
- Las referencias marcadas en el paper como **[M]** (software de compresión: zstd, brotli, bzip2, xz) están citadas de memoria; el resto se verificó por título, autores y venue, sin leer los artículos completos.
- Los costes de GPU del paper son **supuestos**; el coste de entrenamiento en CPU sí está medido.
- Ningún resultado afirma competitividad con modelos de frontera.

## Cómo citar

> laboratório de pesquisa e desenvolvimento. *Ixchel: modelado de lenguaje por mezcla de expertos de conteo, sin gradientes ni hiperparámetros ajustados, en CPU pura.* Preprint v1.0, noviembre de 2026. Repositorio: https://github.com/YSI-Labs/YS-001

## Licencia

- **Código:** [MIT](LICENSE).
- **Paper, figuras y documentación:** [CC BY 4.0](LICENSE-CC-BY-4.0.md).
- **Datos de terceros:** sujetos a sus propias licencias; ver [`docs/DATOS_Y_LICENCIAS.md`](docs/DATOS_Y_LICENCIAS.md).

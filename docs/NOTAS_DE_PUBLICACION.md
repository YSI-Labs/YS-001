# Notas de publicación

Este documento registra qué se cambió en el repositorio respecto al proyecto de trabajo local, para que quien revise el repositorio pueda verificarlo.

## Cambios en el código

- **Rutas absolutas eliminadas.** Los scripts apuntaban a la carpeta de trabajo de su autor (`D:\...\arquitectura-nueva`). Se sustituyeron 45 literales en 39 archivos `.py` por una expresión relativa al propio archivo, que resuelve a la raíz del repositorio. Ejemplo:
  - Antes: `sys.path.insert(0, r"D:\Escitorio - Original\...\arquitectura-nueva\common")`
  - Ahora: `sys.path.insert(0, __import__("os").path.join(<raíz calculada desde __file__>, 'common'))`
  - Los dos scripts `.sh` de `r1_ik` y `r1_ixim` usan `cd "$(dirname "$0")"`.
- **Código no modificado.** Ninguna otra línea de código cambió. Todos los `.py` compilan (`py_compile`).

## Lo que se conserva tal cual

- **Registros de ejecución** (16 archivos `.log`, un `.md` y un `.txt`): conservan rutas absolutas de la máquina original. Son el registro literal de lo que se ejecutó y no se editan. Por ejemplo, `experiments/r3_engine/train_log.txt` y `experiments/BRIEF.md` (instrucciones que se dieron a los agentes).
- **Cifras**: no se ha modificado ningún número de los JSON, tablas o informes.

## Lo que se excluyó

- `data/` completo (≈ 41 GB): corpus de terceros, tablas y muestras.
- Cachés y pesos (`*.npy`, `*.npz`, `*.pt`, `*.safetensors`, `*.bin`, carpetas `cache/`, `model_v03/`, `_test_big/`): ≈ 800 MB, regenerables con los scripts.
- `preprint/PREPRINT_v1_backup.md`: copia de trabajo anterior del paper. La versión vigente es `preprint/PREPRINT.md`.
- `ixchel/results/niche1_docs.json`: contiene 200 documentos de C4 completos. Lo regenera `niche1_quality.py`.
- Archivos `_smoke_*` de pruebas en CPU.
- `__pycache__/`.

## Datos personales y de terceros

- **Correos electrónicos.** Se sustituyeron por `[correo-redactado]` las apariciones de un correo de contacto de una empresa, que estaba en fragmentos de C4 reproducidos por los modelos en las muestras `gen_*.json` (10 apariciones). También se redactó una cadena numérica-alfanumérica que el patrón tomó por correo sin serlo; si esa cadena formaba parte de una muestra, la muestra queda alterada en ese punto.
- **Búsqueda de secretos.** Se buscaron tokens (`gho_`, `ghp_`, `sk-`, `AKIA`), claves privadas y correos. No se encontró ningún secreto del autor.
- **Textos de C4 en las muestras.** Las muestras `gen_*.json` contienen fragmentos de C4 generados por los modelos. Se publican como resultado de los experimentos; no se publican los documentos completos de C4.

## Autoría y licencias

- Autor: laboratório de pesquisa e desenvolvimento (único).
- Código: MIT. Paper, figuras y documentación: CC BY 4.0.
- La versión del paper pasa de «borrador» a **preprint v1.0**. Las referencias marcadas **[M]** en el paper (software de compresión) se indican como tal.

## Tamaño

- Repositorio: ≈ 640 archivos y ≈ 5 MB. El archivo más grande es el PDF del paper (≈ 0.4 MB).

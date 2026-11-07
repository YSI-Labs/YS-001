# BRIEF COMUN - Ronda 1 de MVPs (leer entero antes de empezar)

## Proyecto
Buscamos una arquitectura de modelo de lenguaje NUEVA, que NO sea transformer ni use atencion, que:
- se entrene en CPU pura (sin GPU), en segundos/minutos, con MUCHOS datos;
- NO use AdamW/SGD con learning rate tuneado ni "recetas de hiperparametros": el entrenamiento ideal es
  conteo, solucion cerrada (algebra lineal), o aprendizaje online sin parametros libres;
- escale a millones/miles de millones de parametros solo anadiendo memoria/CPU (sin cuellos de botella tipo backprop);
- genere texto con sentido. Corpus de prueba: obras completas de Shakespeare, nivel caracter (V=96).
Cada experimento de esta ronda es un MVP MINIMO (no un producto): queremos DATOS EMPIRICOS honestos
para decidir que direccion seguir. Un resultado negativo bien documentado es tan valioso como uno positivo.

## Entorno (Windows 11, PowerShell/Git-Bash disponible)
- Python 3.14: usa `python -I script.py` (modo aislado). Paquetes: numpy 2.4, numba 0.65, scipy 1.17 (torch solo para el agente baseline).
- NO instales paquetes (pip). Maquina: Ryzen 7 (8 nucleos/16 hilos) compartida con ~7 agentes mas corriendo a la vez:
  **limita a 2 hilos**: al INICIO de tu script, antes de importar numpy/numba, haz
  `import os; os.environ["NUMBA_NUM_THREADS"]="2"; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"`.
  Los tiempos que midas estan algo contaminados por la competencia; anotalo, pero no pasa nada.
- RAM: max ~3 GB por experimento. Ningun experimento individual debe pasar de ~10 min de pared; evita bucles Python por caracter
  sobre los 4.8M de entrenamiento (usa numpy vectorizado o numba @njit).
- Directorio raiz: `D:\Escitorio - Original\Escritorio\arquitectura-nueva`. Trabaja SOLO dentro de tu carpeta `experiments/r1_<codename>/`.
  NO modifiques `common/` ni otras carpetas. Scripts temporales -> tu misma carpeta.

## Datos y harness (NO reimplementes la evaluacion)
- `common/harness.py`: `load()` -> (train uint8, val uint8, itos); `run(model, n_eval, gen_chars, out_json=...)`, `summarize(res)`.
  Importa asi: `import sys; sys.path.insert(0, r"D:\Escitorio - Original\Escritorio\arquitectura-nueva\common"); import harness`.
- Contrato de modelo (lee la docstring de harness.py): `fit(train, V)`, `predict(ids, pos) -> float32 [len(pos), V]` con
  p(ids[pos] | ids[:pos]) normalizada y > 0, `n_params()`, atributo `name`. `predict` NO puede mirar ids[pos:] (hay leak_check automatico: debe dar leak_free=True).
- `common/baseline_wb.py` es un ejemplo funcional del contrato (n-gram Witten-Bell). Referencia a batir:
  **WB-ngram-5: 2.269 bpc, 1.08M params, fit 0.6 s.** (WB orden 3: 2.483; azar: 6.58; unigrama ~4.7.)
- REGLA DE ORO anti-trampa: el tramo `val` SOLO se usa para el reporte final (harness.run). Si necesitas elegir algo
  (rango, escala, regularizacion...), hazlo con una porcion reservada del propio TRAIN (p.ej. los ultimos 5% de train como "dev") y
  luego reentrena con todo o deja dev fuera, y DOCUMENTALO. No ajustes nada mirando val.
- Muestreo: temperatura 1.0 (el harness ya lo hace). No uses top-k/temperatura para "arreglar" el texto.

## Presupuesto y proceso
- ~40-60 min de trabajo. Maximo 3 iteraciones de mejora tras el primer MVP funcional. Primero un MVP que corra de punta a punta
  (aunque sea malo), luego mejoras. Usa n_eval=20_000 mientras iteras y n_eval=50_000 en la corrida final.
- Se HONESTO: reporta bugs, resultados malos, cosas que no funcionaron y por que. No inventes numeros: todo numero del REPORT
  debe salir de un JSON/log que dejaste en tu carpeta. Si leak_free sale False, arreglalo antes de reportar.
- No uses transformers, atencion, ni backprop/Adam. Si tu idea necesita ajustar parametros por gradiente, usa solo metodos sin
  learning-rate libre (conteo, minimos cuadrados cerrados, apuestas de monedas/"parameter-free online learning", Bayes) y dilo.

## Entregables (en `experiments/r1_<codename>/`)
1. `model.py` (clase del modelo), `run.py` (corrida final reproducible, escribe `result.json` via harness.run(out_json=...)).
2. `REPORT.md` (maximo ~70 lineas, en espanol) con estas secciones:
   - **Idea** (3-5 lineas) y **que implementaste** (que es MVP y que quedo fuera).
   - **Resultados**: tabla con cada variante probada: bpc (val), params, fit_wall_s, chars/s de entrenamiento, ch/s de inferencia. Incluye la muestra de texto generado (300-400 chars) de la mejor variante.
   - **Auditoria de hiperparametros**: lista TODA constante libre (dimension, orden, semilla, escala...) y si la fijaste a mano, la derivaste de datos o es "natural" (p.ej. V). Cuenta cuantas son realmente libres.
   - **Fallas y hallazgos negativos** (lo mas importante para el preprint): que intentaste que no funciono, por que crees que fallo.
   - **Escalabilidad**: como crecen bpc/tiempo/memoria al duplicar el tamano del modelo (mide al menos 2 tamanos si puedes).
   - **Veredicto**: probabilidad 0-100% (justificada) de que esta direccion, con mas datos/tamano, supere a n-gram WB y se acerque a modelos neuronales pequenos; y que harias a continuacion.
3. Tu ultimo mensaje al terminar: resumen de <= 15 lineas con el mejor bpc, params, tiempo y la ruta del REPORT.md.

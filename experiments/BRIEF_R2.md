# BRIEF RONDA 2 - extender Ixchel v0.1 (leer entero)

## Contexto
Ronda 1 (ver `experiments/ROUND1_SUMMARY.md`): los estados densos/aleatorios (reservorio, HDC, rasgos aleatorios) fallaron; lo que funciona es
CONTEO EXACTO + suavizado Kneser-Ney + mezcla informada. Construi el nucleo **Ixchel v0.1** en `ixchel/core.py` (LEELO ENTERO, ~300 lineas):
- ATOMOS (rasgos discretos del pasado: `c1..c12`, clases `k1..k8`, prefijo de palabra `wp`, palabras previas `pw1/pw2`, `lpos`, `wlen`, `lfirst`, `pw1f`, `sp`),
- CADENAS (lista de niveles grueso->fino; cada nivel = lista de atomos) = modelo KN modificado jerarquico (descuentos D1,D2,D3+ en forma cerrada),
- MEZCLA lineal p = sum_k w[g,k] p_k con pesos por EM (punto fijo, sin learning rate) y compuerta g discreta (nivel mas largo visto + conteo). Los pesos se ajustan en un
  tramo dev INTERNO del train (95%/5%) y luego las tablas se recuentan con todo el train. `ixchel/run_v01.py` muestra como correrlo con el harness.
Resultados v0.1 (val, n_eval=20k, `ixchel/results/v01_*.json`): A (contigua) 2.126 | A+W 2.104 | A+W+S 2.079 | A+W+S+L **2.078** (15.6M pares, fit 52 s) | pesos globales (sin compuerta) 2.086.
Referencias: WB-5 2.269; GRU 0.84M params entrenada 8 min en CPU 2.314 (transformer pequeno 3.59). Baseline GRU de 30 min en curso.
Objetivo de la ronda: bajar de 2.078 con ideas que sigan siendo CONTEO / ALGEBRA CERRADA / PUNTO FIJO, sin AdamW ni learning rates libres, y medir cada aporte con ablaciones.

## Reglas (igual que ronda 1; releer `experiments/BRIEF.md` sec. Entorno, Datos y harness, regla anti-trampa)
- No modifiques `ixchel/` ni `common/`. COPIA `ixchel/core.py` a tu carpeta `experiments/r2_<codename>/core.py` y trabaja ahi (y scripts propios en esa carpeta).
- `val` SOLO para el reporte final via harness.run; todo ajuste (incluido el EM/mezcla) se hace con el dev interno del train. No uses val para elegir nada.
- Python: `python -I script.py`; limita hilos a 2: `NUMBA_NUM_THREADS=2` ANTES de importar core (core usa setdefault 4; fija tu env primero) y OMP/MKL=2. Maquina compartida (otra GRU usa 8 hilos): tiempos contaminados, anotalo.
- RAM <= 4 GB por proceso. Cada corrida <= ~10 min. Evita bucles Python por caracter sobre millones de posiciones (numba o numpy).
- Presupuesto ~45-60 min. Primero un resultado funcional, luego mejoras; maximo 3 iteraciones. Mide siempre contra la linea base v0.1 (A+W+S+L = 2.078) con el MISMO n_eval=20000 y deja el JSON en tu carpeta. leak_free debe ser True.
- Honestidad: reporta lo que NO funciono y por que; cada cifra del informe debe salir de un JSON/log tuyo. Audita hiperparametros nuevos (cuales son libres, cuales derivados de datos). Prefiero ideas con CERO constantes libres nuevas.
- NO puedes escribir un archivo llamado REPORT.md (el sistema lo bloquea a subagentes): entrega el informe COMPLETO como tu mensaje final (<= 80 lineas, espanol) con: idea, que implementaste, tabla de ablaciones (bpc val n_eval=20k + params + fit s + ch/s inferencia), fallas/hallazgos negativos, auditoria de hiperparametros, escalabilidad, veredicto (que parte conviene fusionar a Ixchel v0.2 y como: indica el parche exacto o el fragmento de codigo clave y la ruta del archivo donde esta). Guarda tu codigo y JSON en tu carpeta.

# REPORT r2_classes (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/r2_<variante>_n20000.json y logs/)

**Veredicto: la generalizacion por clusters de palabras NO ayuda; las clases de caracter solo ayudan con k=4 (-0.013 bpc), pendiente de confirmar.**

Idea: dar a los expertos de conteo generalizacion entre contextos parecidos, sin gradientes: (1) clusters de palabras (top-2000 palabras, PPMI + SVD d=32 + k-means, k=round(sqrt(N))=45, raras por forma: ultimas letras), atomos pc1/pc2; (2) clases de caracter aprendidas por conteo (PPMI+SVD d=16 + k-means; rejilla {4,8,16,32,64}; k "auto" por BIC) en lugar de las 8 clases a mano de la cadena S. (3) embedding-kNN: no implementado.

## Resultados (val, n_eval=20000; leak_free=True; fit con 2 hilos en maquina compartida, tiempos contaminados)
| variante | bpc val mezcla | params | fit s | bpc dev cadena S | bpc dev mezcla |
|---|---|---|---|---|---|
| base AWSL (control v0.1) | 2.0784 | 15.65M | 72 | 2.2187 | 1.9878 |
| pcA: C=[pc1],[pc1,pc2] | 2.0783 | 15.85M | 103 | | 1.9878 |
| pcB: C con wp,pc1,pc2,c1 | 2.0775 | 19.75M | 137 | | 1.9872 |
| pcC: C con pc1,pc2,c1,c2 | 2.0776 | 17.54M | 129 | | 1.9871 |
| **c4 (clases aprendidas)** | **2.0651** | 14.88M | 85 | 2.2091 | **1.9791** |
| c8 | 2.0747 | 15.82M | 93 | 2.1981 | 1.9898 |
| c16 | 2.0814 | 16.66M | 93 | 2.1932 | 1.9924 |
| c32 | 2.0839 | 17.45M | 98 | 2.1902 | 2.0055 |
| cAuto (BIC) | 2.0911 | 18.42M | 78 | 2.1822 | 2.0151 |

## Hallazgos
- Clusters de palabras: sin efecto (cobertura 84% de tokens; k-means converge en 19 iteraciones); la mezcla no les da peso (cadena C aislada 3.4-4.7 bpc en dev).
- Clases de caracter: en la cadena S aislada mas clases mejora monotonamente (2.22 -> 2.18), pero en la MEZCLA no es monotono (c4 gana, c32 empeora): la ganancia de la cadena no se traslada a la mezcla (redundancia con las cadenas contiguas).
- Regla BIC fallida: elige el maximo de la rejilla (k=64); la penalizacion 0.5*(k^2+V)*log2(N) es demasiado debil con 4.8M tokens.
- Costo: agrupar palabras anade ~30-60 s de fit y 0.2-4M params; clases de caracter, pocos segundos.
- c4 NO fue elegida por val: la rejilla se fijo antes y el dev tambien la prefiere (1.9791 vs 1.9878). Confirmar con otra semilla o tramo dev antes de fusionar (ganancia 0.013 ~ 1 error estandar de la evaluacion).

Constantes libres introducidas (>=6): N=2000, d=32, d=16, rejilla {4,8,16,32,64}, semilla 0, tope 100 iteraciones k-means, 32 buckets de forma. k de palabras derivado por sqrt(N); k de caracteres por rejilla/BIC (BIC fallo).

Fusion a v0.2: solo c4, tras confirmacion. No fusionar clusters de palabras.

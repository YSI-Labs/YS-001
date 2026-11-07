# REPORT r1_chan (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de result_proto.json, result_wb.json, sweep_*.log, diag_norm.json de esta carpeta)

**Veredicto: NEGATIVO.** Computacion hiperdimensional densa (HDC) no se acerca a WB-5. Mejor val: 3.870 bpc (variante `wb`) vs 2.269.

## Idea e implementacion (MVP)
Hipervectores bipolares fijos por caracter; contexto ligado por permutacion ciclica y producto (XOR) para n-gramas; memoria Hebbian = sumas enteras, sin entrenamiento iterativo. Dos modos:
- `proto`: un prototipo por caracter siguiente, M_c = suma de C(t) (superposicion de ordenes 1..K); puntaje <C,M_c>/(D*N); p = softmax(beta*z).
- `wb` (arreglo): memoria por orden con conteos HDC particionada por hash de (x[t-1],x[t-2]) mod H; conteo estimado s=<K_k,M>/D; mezcla recursiva tipo Witten-Bell con cnt=max(s - z*sigma, 0), sigma^2 = N_bucket,c/D (ruido de superposicion conocido).
Tuning solo en dev (ultimo 5% train). NO implementado: SDM/cleanup memory, olvido selectivo, mezcla con n-grama bajo.

## Resultados (val, 50k posiciones, leak_free=True en ambas)
| variante | bpc val | params (celdas) | fit s | train ch/s | inf ch/s |
|---|---|---|---|---|---|
| WB-5 (ref) | 2.269 | 1.08M | 0.6 | 8.1M | ~28k |
| Chan proto D4096 K8, beta=239.9 | 5.043 | 393k | 48.7 | 99k | 8.8k |
| Chan wb D2048 K5 H31 z=4 | 3.870 | 30.5M | 86.5 | 55.8k | 1.4k |
Dev2 (ref WB-5 = 2.296): proto D4096 K=1/2/3/4/6/8: 5.27/5.12/5.08/5.06/5.05/5.06; proto K8 D=1024/4096: 5.09/5.06; wb sin particion (H=1) K5 D=1024/2048/4096: 7.00/6.40/5.86; wb D2048 K5 H=1 vs H=31: 6.40 vs 3.84; z=0..4 (H31): 3.775/3.766/3.755/3.749/3.740; proto K3 softmax vs relu-normalizado: 5.20 vs 3.73; proto K8: 5.17 vs 3.87.
Muestra (wb): casi ruido de caracteres, sin palabras ni gramatica.

## Hiperparametros
Libres significativas ~5: D, K, beta (1-D en dev1), H (solo {1,31}), z (en el borde del grid 0..4). Menores: semilla, umbral 0.5, suelo U>=1, +1 unigrama.

## Fallas / hallazgos negativos
1. `proto` no esta normalizado por la masa del contexto: colapsa a ~unigrama (5.05); normalizar con relu(z)/sum baja de 5.2 a 3.7 en dev. D y K no lo arreglan.
2. Ruido de superposicion: desviacion sqrt(N_c/D) en conteos (para 'e', N_c~4.6e5 con D=4096: ~10 conteos) entierra contextos de orden >=3 (conteos 1-30). Mas D mejora monotonamente (7.00->6.40->5.86), como predice sqrt(1/D).
3. Particionar por hash de 2 caracteres previos (H=31) es la mejora real (6.40 -> 3.84); el umbral de ruido aporta poco.
4. Brecha restante ~1.6 bpc (3.87 vs 2.27): HDC denso es un sketch aleatorio de tablas de conteo; para sigma<1 hace falta D*H >~ N_c, o sea celdas proporcionales a los datos. WB usa 1.05M entradas contra 30.5M celdas para mejor bpc.
5. Nota: result_wb.json dice "z0.0" en el nombre; el z efectivo es 4.0 (campo tuning).
6. Recortes: sweep2 se detuvo tras 3 de 7 configs; proto sin D=16384; relu-normalizado medido solo en dev2, no en val. CPU compartida: tiempos inflados.

## Escalabilidad
proto: D 1024->4096 (x4 params): dev2 5.09->5.06, no escala. wb H=1: D 1024/2048/4096: 7.00/6.40/5.86; fit 16.7/29/41 s. Memoria K*V*D*4 B (122 MB para H31 D2048 K5), lineal en D y H. Cuello: bucle Python por bucket en inferencia.

## Veredicto del agente
~3%. El error de conteo crece como sqrt(N/(D*H)); bajarlo exige celdas al ritmo de los datos: un sketch de la tabla exacta de n-gramas, que ya es mas compacta. Si se insiste: normalizar por masa en proto + WB bajo orden; SDM con N_loc del orden de contextos distintos; no invertir mas en D denso.

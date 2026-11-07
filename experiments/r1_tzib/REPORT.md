# REPORT r1_tzib (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de los result_*.json y logs de esta carpeta)

**Veredicto: unico candidato que mejora el baseline.** Kneser-Ney con descuentos en forma cerrada y orden 5: **2.185 bpc (n_eval 20k) / 2.195 (50k)** vs WB-5 2.269 (2.281 en 50k). Orden ilimitado NO ayuda: memoriza.

## Idea e implementacion
LM de orden no acotado sin entrenamiento por gradiente. Arreglo de sufijos sobre el train invertido (4.82M chars), extension de contexto = refinamiento de intervalo (numba). Mezcla las distribuciones de siguiente-caracter de TODAS las longitudes de coincidencia 1..L* con pesos cerrados: Witten-Bell (WB), Kneser-Ney con D = n1/(n1+2 n2) (derivado de los datos), o WB con exclusion (EXCL).
Verificado: WB-5 == common/baseline_wb.py (diff 0.0 en 300 posiciones); KN-5 y EXCL-5 coinciden con calculo independiente por fuerza bruta (<=3e-8); leak_free=True siempre. Fuera de alcance: SA-IS, compresion del indice, tuning en val.

## Resultados (val bpc, n_eval=20k)
| variante | bpc | params (celdas de conteo) | eval ch/s |
|---|---|---|---|
| WB-5 | 2.269 | 406,064 | 738 |
| WB-12 | 3.002 | 17.3M | 1,597 |
| WB-20 | 3.112 | 53.6M | 1,401 |
| WB-inf | 3.123 | 72.7M | 2,069 |
| **KN-5** | **2.185** | 406,064 | 1,871 |
| KN-12 | 2.314 | 17.3M | 1,921 |
| KN-20 | 2.319 | 53.6M | 1,909 |
| KN-inf | 2.318 | 72.7M | 1,973 |
| EXCL-5 | 2.204 | 406,064 | 2,478 |
| EXCL-inf | 2.434 | 72.7M | 2,460 |
n_eval=50k: KN-5 2.195, WB-5 2.281. Construccion del indice en frio ~36-41 s (SA ~23-26 s), ~120k ch/s.
Muestra (KN-5): "...DUKE.\nHath head\nTo shrice thou happy husband on my heard was time\nI know he's as a very put them sprays;\nTurns that's no pois'd! How how friends..."

## Copia literal (3000 chars generados; segmento mas largo copiable del train)
KN-5: segmento medio 9.1 chars (referencia texto humano val: 10.2; 0.8% de la salida en segmentos >=20). WB-inf: 2 segmentos, 100% de la salida en segmentos >=20, max 2000 chars => reproduce el train verbatim (memoriza).

## Hiperparametros
Libres: topes de orden 5/12/20 (dados por el brief), semilla 0, constantes de implementacion DMAX=24, LBUF=4096 (sin truncamiento observado). D de KN = formula cerrada del train. Cero learning rates, cero valores ajustados en val.

## Fallas
1. Primer KN daba probabilidades cero (D[0] salia de conteos unigrama, n1=0 => D=0); arreglado con count-of-counts del k-grama para niveles crudos y del (k+1)-grama para niveles de continuacion. Esa asignacion es eleccion del agente, no teoria.
2. WB ilimitado: cada nivel extra de conteo 1 da 50% de masa al mismo caracter => sobreconfianza y memorizacion.
3. KN ilimitado es menos malo que WB pero peor que tope 5. Exclusion a tope 5 cerca de KN-5 (2.204); ilimitada es mala (2.434).

## Escalabilidad (20k pos, fraccion de train 25/50/100%)
KN-5: 2.376/2.294/2.185; WB-inf: 3.285/3.215/3.123; KN-inf: 2.505/2.431/2.318. Duplicar datos: ~-0.08 a -0.09 bpc (KN-5). Celdas ~x1.3 por duplicacion. Memoria indice: SA ~39 MB int64 + ~15 MB tablas densas.

## Veredicto del agente
Orden ilimitado + WB sobreajusta y memoriza (confirmado). KN cerrado orden 5 supera a WB-5 con cero parametros aprendidos (confirmado). ~85% de que la direccion supere a WB-n con mas datos; ~5% de acercarse a redes pequenas (~1.5-1.7 bpc aqui). Siguientes pasos: KN a orden 5-7 con descuentos por nivel medidos en dev del train; mezcla parameter-free entre ordenes KN (Bayes/ponderacion por conteos); probar en corpus multiautor mayor.

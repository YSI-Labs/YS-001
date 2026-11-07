# REPORT r3_longcache (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/lc_*.json y compare_*.json)
**Veredicto: fusionar el cache de historial de ORDEN HASTA 8 (4 ventanas) en lugar del CacheWB de orden 4.** Unica mejora consistente en text8 y Shakespeare.
Idea: KeyedWin = cadena Witten-Bell jerarquica sobre claves ('c',k) k<=8 | ('w',) prefijo de palabra | ('w2',) (pw1,wp), contada SOLO en el historial (ventana dura o decaimiento exponencial exacto). Indice construido una vez; consultas causales t<p.
## text8 (train 3M, n_eval 20000; base = 'full' = 1.6783, leak_free=True) delta vs base (+ = mejor), SE pareado por bloques
| variante | bpc | delta | SE bloques |
|---|---|---|---|
| (1) palabras wp y (pw1,wp), ventanas 1k/4k/16k/inf | 1.6525 | +0.0258 | 0.0017 |
| (3) tema: wp, ventanas 2000/5000 | 1.6702 | +0.0081 | 0.0008 |
| (2) decaimiento exp. orden 1-4 (tau dev-ML=5525) | 1.6804 | -0.0021 | 0.0008 |
| (4) cache orden 6 (reemplaza WB<=4) | 1.6428 | +0.0356 | 0.0016 |
| **(4) cache orden 8 (reemplaza WB<=4)** | **1.6300** | **+0.0484** | 0.0022 |
| orden 6 + palabras | 1.6350 | +0.0433 | 0.0021 |
Fraccion de posiciones donde gana a la base: palabras .59, tema .55, decaimiento .39, orden 6 .67, orden 8 .69.
## Shakespeare (train 4.8M, n_eval 10000; base = v0.3 = 1.9233)
orden 8: 1.9125 (+0.0108, SE 0.0021); palabras encima de la base: 1.9209 (+0.0024, SE 0.0013).
## Hallazgos / fallas
- Palabras y tema se SOLAPAN con el orden alto (palabras: +0.026 solo, +0.008 sobre orden 6). Decaimiento exponencial EMPEORA (tau elegido sobre dev de 150k no transfiere a historiales de millones). Tema +0.008 solapado: descartado.
- Primer intento de decaimiento con historial train[:p] en dev sesgaba los pesos (el experto veia datos ya contados por las cadenas); corregido (mismo historial que el cache base).
- PoE con 15-17 expertos era el cuello: reescrito con BLAS (equivalencia verificada, dif <1e-6 en w, test_mixing_equiv.py).
- Costo: inferencia 1330 -> ~510 ch/s con orden 8 (indice por llamada; falta indice incremental exacto: ver r3_engine). Indice orden 8 en val 5M: ~0.5 GB.
- No medido: orden 8 + palabras en text8; ventanas largas con dev corto (150k) vs test (hasta 5M).
- Proceso: durante una limpieza el agente mato bash.exe por patron amplio ("run_lc"); no verifico si alguno era de otro agente.
## Hiperparametros
A priori: ordenes {6,8}; ventanas {256,1024,4096,inf}; palabras {1k,4k,16k,inf}; tema {2000,5000}. Estimado en dev: tau (1 continuo). Numericos: ridge 1e-6, tol 1e-7. ~6 decisiones de diseno.

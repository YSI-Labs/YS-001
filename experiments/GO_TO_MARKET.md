# GO_TO_MARKET (honesto): que se puede vender hoy y que no (2026-11-06)

Etiquetas: **[MEDIDO]** sale de nuestros JSON; **[HIPOTESIS]** idea con prueba pendiente; **[M]** dato de memoria no verificado (verificar antes de publicar). Los precios y mercados no se investigaron aqui.

## 1. Veredicto
Ixchel v0.5 NO puede venderse como LLM general ni como agente: **[MEDIDO]** HumanEval 0/164; HellaSwag 28.8 % (azar 25), PIQA 56.5 % (azar 50), ARC-Easy 30.1 % (azar 25); texto generado a T=1 es ininteligible (Qwen lo puntua a 3.2 bits/car frente a 0.83 del texto real) y a T<1 cae en bucles. Esto se mantuvo al pasar de 54M a 1,17 mil millones de celdas y de 85M a 656M caracteres: no es un problema de tamano, es que el conteo exacto no comparte informacion entre contextos distintos. Vender tokens de este modelo como si fueran de un LLM seria enganoso.

## 2. Lo que Ixchel hace bien **[MEDIDO]**
- Entrena SIN optimizador ni tasa de aprendizaje: 89 min de pared para 1,17 mil millones de celdas en CPU de escritorio (frente a ~19 anos de CPU para entrenar una red de 1B con 20B tokens, `REAL1B_REFERENCE.md`).
- Inferencia barata: ~5k caracteres/s por hilo; 9.2k posiciones/s en la evaluacion grande; sin GPU.
- Da una probabilidad calibrada por caracter (bpc 1.3 en codigo, 1.4 en mates, 1.6 en web): sirve para *puntuar* texto, no para *entenderlo*.
- Es incremental: anadir datos significa sumar conteos, sin reentrenar desde cero (propiedad de la arquitectura; el bit-identico de la construccion fuera de memoria se verifico a 1.5M de caracteres).

## 3. Nichos candidatos (todos **[HIPOTESIS]**, con su prueba)
| # | uso | por que podria funcionar | prueba que lo decide | riesgo |
|---|---|---|---|---|
| 1 | Puntuacion/filtrado de calidad de datos para preentrenar | solo necesita probabilidades baratas y se entrena en horas; el filtrado por perplejidad de un n-grama es practica conocida [M] | puntuar un corpus con etiquetas de calidad conocidas y medir AUC; luego entrenar dos redes pequenas, con y sin el filtro, y comparar | un n-grama KN ya hace esto; hay que demostrar que Ixchel (cadenas + cache) puntua mejor que un n-grama simple |
| 2 | Autocompletado local de codigo/texto (sugerencia de siguiente fragmento) | decodificacion de microsegundos en CPU, privado, sin GPU | medir caracteres ahorrados por tecla y tasa de aceptacion sobre repos no vistos, frente a un baseline n-grama y frente a un modelo pequeno | top-1 por caracter (73 % en codigo) no implica sugerencias utiles; sin semantica falla en codigo nuevo |
| 3 | Borrador (drafter) para decodificacion especulativa de un LLM grande | verificar un borrador barato puede acelerar el modelo grande; los borradores basados en n-gramas existen [M] | reconstruir Ixchel a nivel de TOKENS del LLM (hoy es a nivel de caracter, tokenizador incompatible), medir tasa de aceptacion y aceleracion real | requiere rehacer tablas sobre otro vocabulario; si la aceptacion es baja no compensa |
| 4 | Compresion de texto/codigo con codificacion aritmetica | el bpc medido (1.3-1.6) es el tamano comprimido ideal; 656M de caracteres entrenados | comprimir el val con Ixchel + codificador aritmetico y comparar bytes contra xz/zstd sobre los MISMOS archivos (no medido todavia) | las tablas pesan 17 GB (hay que contarlas en el tamano) y 9k car/s es lento |
Prioridad sugerida: 1 (menor costo de prueba, 1-2 semanas), luego 4, luego 3. El 2 solo si el 1 sale bien.

### 3.1 RESULTADO de la prueba del nicho 1 (2026-11-06; `ixchel/niche1_*.py`, `results/niche1_analysis.md`) **[MEDIDO]**
Diseno y regla de decision fijados antes de ver datos (cabecera de `niche1_quality.py`). 500 documentos de un shard de C4 que el entrenamiento no uso, 1000 caracteres cada uno. Puntuador = bits por caracter (mayor = peor). Comparados: Ixchel v0.5 completo (big2), n-grama KN simple (cadena A del mismo modelo, mismos datos) y tasa de compresion zlib.
E1, AUC limpio vs degradado (IC95% por documento): ruido 2 % 0.940 / **0.950** / 0.635; ruido 10 % 0.998 / 0.997 / 0.934; palabras barajadas 0.974 / **0.984** / 0.623; **repeticion 0.000 / 0.796 / 0.000**; empalme de dos docs **0.629** / 0.555 / **0.701** (orden: Ixchel / KN / zlib).
E2, Spearman con el bpc de Qwen2.5-1.5B (200 docs): Ixchel 0.687 [0.59, 0.76]; KN 0.592 [0.48, 0.69]; zlib 0.437 [0.30, 0.56]. Diferencia pareada Ixchel-KN **+0.095 [0.043, 0.150]** (analisis POST HOC, no estaba en el diseno).
Lectura honesta:
- Segun la regla fijada (Ixchel completo debe superar a KN con IC no solapado en la mayoria de degradaciones), **no se cumple**: en ruido y palabras barajadas empatan o gana KN por ~0.01; Ixchel solo gana en el empalme (0.63 vs 0.56) y, post hoc, en la correlacion con Qwen.
- **Falla grave:** la cache de historial hace que Ixchel puntue el texto repetido como el mas "limpio" (AUC 0.000; bpc 0.17 vs 1.70). Un filtro de calidad de datos por bpc dejaria pasar justo el texto repetido/boilerplate. KN solo lo detecta de forma parcial (0.80); zlib falla igual que Ixchel. Habria que filtrar la repeticion con una regla aparte.
- Ninguno detecta bien el empalme de dos documentos (<= 0.70): no entienden la coherencia.
- Rapidez: ~18 mil caracteres/s de puntuacion (3500 docs en ~3 min) con 8 hilos; Qwen2.5-1.5B tardo 640 s para 200 docs (~300 car/s) en la misma maquina: unas 60 veces mas lento.
- Limites: degradaciones sinteticas; E2 usa la perplejidad de Qwen como proxy (no es calidad humana); n=200; C4 ya esta filtrado; no se midio el efecto en un modelo entrenado con/sin filtro.
**Veredicto del nicho 1: hipotesis NO confirmada para Ixchel completo; un n-grama KN barato (ya parte del modelo) puntua casi igual en degradaciones y peor solo en la correlacion con un LM fuerte. Sin la prueba de entrenar con/sin filtro no hay evidencia de que valga la pena el modelo de 17 GB frente a un n-grama de unos pocos GB.**

### 3.2 Seguimiento del nicho 1: Ixchel SIN cache y detector de repeticion (`ixchel/niche1_nocache*.py`, `results/niche1_nocache_analysis.md`) **[MEDIDO]**
Mismos 500 docs y degradaciones. Regla fijada antes: "sin cache" queda corregido si su AUC en `repeat` > 0.5 con IC por encima de 0.5.
AUC (IC95%) con cache / sin cache / KN (A): ruido 2 % 0.940 / 0.943 / 0.950; palabras barajadas 0.974 / 0.978 / 0.984; **repeticion 0.000 / 0.806 [0.784, 0.827] / 0.796**; empalme 0.629 / 0.551 / 0.555. Spearman con Qwen: 0.687 / 0.612 [0.50, 0.70] / 0.592. Detector `gap` = bpc sin cache - bpc con cache: AUC **1.000** en repeticion (sintetica: 80 caracteres repetidos), pero 0.64-0.88 en ruido/barajado y 0.32 en empalme (se activa tambien con otras degradaciones). El puntuador combinado (maximo de rangos de bpc sin cache y gap, sin constantes) rinde 0.916 en repeticion pero cae en ruido 2 % (0.796) y empalme (0.401): no sirve como puntuador unico.
Lectura: (i) la falla de repeticion se corrige quitando la cache (0.806) o con `gap` como detector aparte; (ii) **Ixchel sin cache no se distingue de KN**: todos los IC se solapan; la cadena A sola pesa ~3.3 GB (19 % de los pares; estimacion proporcional) frente a 16.9 GB del modelo; (iii) la cache aporta la sensibilidad al empalme (0.63 vs 0.55) y parte de la correlacion con Qwen (+0.07), mecanismo no investigado. **Veredicto del nicho 1: queda como "un n-grama KN de pocos GB basta para puntuar degradaciones sinteticas"; Ixchel grande no aporta una ventaja demostrada. La prueba de entrenar con y sin filtro sigue pendiente.**

### 3.3 RESULTADO del nicho 4: compresion (`ixchel/niche4_compress.py`, `results/niche4_compress.json`) **[MEDIDO]**
Longitud de codigo IDEAL (cota de Shannon, -log2 p causal sobre el archivo completo de validacion; NO se implemento el codificador aritmetico, que anadiria una sobrecarga pequena [M] y exige determinismo numerico entre maquinas) de Ixchel v0.5 (big2) frente a compresores estandar sobre los mismos archivos (val_*, ~0.5-1 M de caracteres cada uno, no usados en el entrenamiento). Zstd con diccionario entrenado con datos del train (112 KB y 1 MB) como alternativa "con conocimiento previo".
| dominio | Ixchel (bytes, bpc) | mejor estandar (bytes, bpc) | cual | razon | xz 9e (bpc) |
|---|---|---|---|---|---|
| web | 200,012 (1.59) | 308,072 (2.45) | zstd-19 + dic. 1 MB | **0.65** | 2.56 |
| codigo | 167,247 (1.31) | 200,746 (1.57) | brotli 11 | **0.83** | 1.60 |
| matematicas | 130,440 (1.43) | 183,010 (2.00) | bzip2 9 | **0.71** | 2.13 |
| chat | 113,173 (1.43) | 184,407 (2.33) | zstd-19 + dic. 1 MB | **0.61** | 2.49 |
| libros | 139,418 (2.08) | 167,167 (2.50) | bzip2 9 | **0.83** | 2.81 |
| total | 750,290 | 1,043,402 | - | **0.72** | - |
Ixchel ahorra 17-39 % frente al mejor compresor estandar de cada archivo. **Pero:** (1) el modelo pesa 16.9 GB; si cada descompresor lo necesita, el equilibrio llega tras comprimir ~160-560 GB de texto similar (web 169 GB, chat 162 GB, mates 253 GB, libros 351 GB, codigo 555 GB; ahorro de 0.03-0.11 bytes/caracter frente a los 18.2e9 bytes del modelo); (2) velocidad ~16-20 mil caracteres/s frente a ~1-2 MB/s de los compresores estandar (los 7 metodos sobre un archivo de 1 MB tardaron 2-3 s en total): ~100 veces mas lento; (3) los compresores de mezcla de contextos (paq8/cmix) o de redes son los verdaderos competidores y NO se probaron; para referencia, Qwen2.5-1.5B mide 0.93 bpc en web (segmentos distintos, `REAL1B_REFERENCE.md`), mucho mejor que Ixchel; (4) la ventaja de Ixchel viene de haber visto 656 M de caracteres de la misma distribucion, mientras que los compresores estandar solo ven el archivo (el diccionario de 1 MB atenua esto en web y chat).
**Veredicto del nicho 4: la ganancia de compresion es real pero la economia solo cierra para un servicio central que comprima cientos de GB de texto de estos dominios y con latencia lenta aceptable; sin comparar contra cmix/paq8 o redes no hay argumento de producto.**

### 3.4 Seguimiento del nicho 4: PPMd y compresor real con ida y vuelta (`ixchel/niche4_ppmd.py`, `niche4_coder.py`) **[MEDIDO]**
PPMd (variante H de 7-Zip, pyppmd, mejor de los órdenes 8, 16 y 32, 256 MB) sobre los mismos archivos completos: web 270,013; codigo 182,803; matematicas 161,688; chat 165,447; libros 152,278 bytes (el orden 16 es el mejor salvo en libros, donde gana el 8 por 196 bytes). Es el competidor mas fuerte probado y mejor que xz y zstd en todos los dominios. Razon Ixchel/PPMd: web 0.74, codigo 0.91, matematicas 0.81, chat 0.68, libros 0.91, es decir, Ixchel ahorra **8-32 %** (no 17-39 %). Siguen sin probarse paq8/cmix y las redes.
Compresor real (codificador aritmetico entero de 32 bits, frecuencias cuantizadas a 65536; extractos de 20,000 caracteres de val_web y val_code): **la ida y vuelta es exacta** (decodificado identico) y la sobrecarga frente a la cota ideal es **+0.05 % (web) y -0.06 % (codigo)**, es decir, nula. Pero el codificador y el decodificador predicen UN caracter por llamada: **44-54 caracteres/s** (con otros procesos compitiendo por CPU) frente a ~16-20 mil caracteres/s del calculo por lotes, ~300 veces mas lento; y el decodificador debe reproducir EXACTAMENTE las mismas probabilidades que el codificador: aqui se probo solo en el mismo proceso, no entre maquinas, hilos o versiones (un solo bit distinto en una probabilidad desincroniza el archivo). **Veredicto del nicho 4: la cota de compresion se alcanza en la practica, pero el servicio es lento y fragil frente a PPMd; sin comparar con paq8/cmix/redes y sin resolver el determinismo no hay producto.**

### 3.5 RESULTADO de entrenar con y sin filtro (nicho 1, la prueba decisiva) (`ixchel/filter_*.py`, `results/filter_analysis.md`) **[MEDIDO]**
Diseno: pool de 8000 docs de C4 (shard sin usar por el entrenamiento de Ixchel), 1000 caracteres cada uno; 400 docs distintos para evaluar. Se entrena una GRU de 0.84M parametros (AdamW lr 3e-3 coseno: es un control estandar, no tecnologia propia) con 1200 pasos x lote 32 x ctx 128 (mismo computo en todos los brazos), 2 semillas. Pool "contaminado": 30 % de los docs sustituidos por una version degradada (ruido 10 %, palabras barajadas, repeticion, empalme). Filtros: quitar el 30 % peor por bpc del n-grama KN (cadena A) o por bpc de Ixchel completo (con cache); la fraccion 30 % = la tasa real de basura, un dato que un pipeline real no conoce.
| brazo | bpc medio (2 semillas) | dispersion entre semillas |
|---|---|---|
| limpio, todos (8000 docs) | 2.2130 | 0.009 |
| limpio, 50 % al azar | 2.2168 | 0.002 |
| limpio, 50 % de menor bpc | 2.2372 | 0.001 |
| limpio, 50 % de mayor bpc | 2.2300 | 0.001 |
| contaminado, sin filtrar | 2.2416 | 0.004 |
| contaminado, filtro KN | 2.2226 | 0.011 |
| contaminado, filtro Ixchel completo | 2.2287 | 0.010 |
| contaminado, quitando los degradados reales (oraculo) | 2.2185 | 0.008 |
Diferencias pareadas por documento (media de 2 semillas, SE por documento; el orden entre brazos se repite en las dos semillas): basura sin filtrar - oraculo **+0.0231 (0.0013)**; filtro KN - sin filtrar **-0.0191 (0.0026)**; filtro Ixchel - sin filtrar -0.0130 (0.0019); filtro KN - oraculo +0.0040 (0.0021); filtro Ixchel - oraculo +0.0101 (0.0014); **filtro KN - filtro Ixchel -0.0061 (0.0016)** (KN mejor). Sobre datos limpios: 50 % de menor bpc - 50 % al azar +0.0204 (0.0032) y 50 % de mayor bpc - 50 % al azar +0.0131 (0.0027): ambos peores que al azar.
Que basura se escapa (de 2400 degradados, el filtro mantiene): KN: ruido 0/600, barajado 2/618, repeticion 254/629, empalme 482/553; Ixchel con cache: ruido 0/600, barajado 9/618, **repeticion 629/629**, empalme 388/553; ademas descarta 738 (KN) y 1026 (Ixchel) docs limpios por completar el 30 %.
Lectura honesta:
- Filtrar basura SI mejora al modelo entrenado: el filtro KN recupera ~83 % del dano de los datos degradados (0.019 de 0.023 bits/car); el efecto es pequeno en valor absoluto y probado con una GRU diminuta, 2 semillas, un tipo de basura sintetica.
- **El n-grama KN barato supera a Ixchel completo** (-0.006 +-0.002) porque la cache de Ixchel deja pasar toda la repeticion. El modelo de 16.9 GB no tiene ventaja demostrada sobre una cadena KN de ~3.3 GB (estimacion proporcional) para este uso.
- Sobre datos limpios, filtrar por perplejidad (quedarse con lo mas facil o con lo mas dificil) empeora frente al azar: el filtro solo ayuda cuando hay basura real que quitar.
- Limites: basura sintetica; la tasa de basura conocida; un solo corpus (C4); GRU de 0.84M y 4.9 M de caracteres vistos, no un LLM; 2 semillas (la dispersion entre semillas, hasta 0.011, es comparable a algunas diferencias: solo se afirman las que superan ese margen y se repiten en ambas semillas).
**Veredicto final del nicho 1: filtrar datos por bpc tiene un efecto real y pequeno, pero lo logra un n-grama KN barato; Ixchel grande no aporta. No es un producto que justifique el modelo de 17 GB; si acaso, una utilidad de preparacion de datos basada en KN.**

## 4. Camino a un producto "vender tokens": red de pesos reales en GPU
Ixchel no es ese producto; la via es una red de pesos reales (costos en `REAL1B_REFERENCE.md`, todos **[SUPUESTO]**). Plan por etapas con puertas de decision:
1. **Piloto de 125M-350M (coste ideal ~2-30 USD por corrida con los supuestos de `gpu_pilot/README.md`; ~100-500 USD como tope prudente con fallos, reintentos y datos; supuesto)**; scripts listos y probados en CPU (no en CUDA) en `gpu_pilot/`: reproducir un transformer pequeno con AdamW como CONTROL, y entrenar la misma red con el optimizador sin tasa de aprendizaje (DoG/Prodigy). Puerta: si DoG queda a >0.05 de perdida del control, aceptar AdamW para el modelo comercial (en pequeno ya iban por detras: ronda 3, `r3_lrfree`). La regla "sin AdamW" del usuario se decide con este dato, no por principio.
2. **1B con 20-100B tokens (~185-920 USD, supuesto)**: medir contra TinyLlama/Qwen con la misma bateria (ya montada: `eval_ref_model.py`). Puerta: HellaSwag >55 % y HumanEval >5 % antes de seguir.
3. **7B**: solo si 1B alcanza las puertas y hay financiacion (~62,000 USD solo de computo ideal, supuesto) y datos de licencia limpia.
4. **Servicio**: la inferencia de un 7B requiere GPU; los precios por token de modelos abiertos ya alojados son bajos y competitivos (no verificado aqui): hay que cotizarlos antes de fijar un margen.
Datos: usar conjuntos abiertos de licencia verificada; no se descargo ni evaluo ninguno de ellos en este proyecto (solo C4 shard 0-3, GSM8K, Dolly y codigo Python local). Revisar licencias y terminos antes de entrenar un modelo comercial.
Papel de Ixchel en ese camino (**[HIPOTESIS]**): memoria/cache de n-gramas exacta junto a la red (idea tipo modelo hibrido; se llamaria con un nuevo nombre maya, p. ej. "Itzamna"). La prueba: mezclar las probabilidades de Ixchel con las de una red pequena entrenada y medir bpc pareado; en la ronda 3 un residuo neuronal sobre Ixchel NO supero al ensamble con una GRU (`r3_residual`), asi que esto no esta probado.

## 5. Lo que NO afirmar
- Que Ixchel "compite con modelos de frontera" o "razona/programa": medido en contra.
- Costos de GPU como datos: son supuestos hasta cotizar y medir un piloto.
- Que un hibrido funciona: no hay ningun experimento que lo respalde todavia.

## 6. Proximo paso concreto recomendado
Los nichos 1 y 4 ya se probaron (3.1-3.5): el 1 queda como utilidad KN barata (Ixchel no aporta); el 4 alcanza la cota pero es lento, fragil y solo 8-32 % mejor que PPMd. Pendiente: (a) repetir el filtrado con un LM mayor, otro corpus y basura natural en lugar de sintetica, y sin conocer la tasa de basura; (b) para el nicho 4: comparar con paq8/cmix o una red pequena y probar el determinismo entre maquinas; (c) nichos 2 y 3 sin tocar. Con estos resultados la conclusion de producto sigue siendo la de la seccion 4: la via comercial es una red de pesos reales en GPU, no Ixchel.

# REPORT r3_lrfree (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de result_*_s0.json y summary_*.json)

**Veredicto: ningun optimizador sin tasa de aprendizaje iguala a AdamW afinado en 2600 pasos; DoG es el unico cercano (+0.099 bpc).**

Idea: entrenar la GRU de referencia (d=256, 2 capas, ctx 128, batch 64, 2600 pasos, semilla 0, mismo orden de minibatches que el baseline para test pareado) sin AdamW, sin schedule, sin weight decay ni clip, con momento 0.9 como unica constante fija.
- **DoG** (Distance over Gradients): eta = rbar / sqrt(sum ||g||^2), rbar = max ||x - x0||, rbar_0 = 1e-6 (1+||x0||), escalar global, momento EMA con correccion de sesgo.
- **Prodigy** (forma simple tipo Adam): d escalar global monotono, d_{k+1} = max(d_k, r/||s||_1).
- **CoinKT** (por coordenada): apuesta KT con moneda recortada; riqueza inicial W0=1e-3.

## Resultados (val n_eval=10000; semilla 0; 2600 pasos; leak_free=True; sin divergencias)
| metodo | bpc val | diferencia vs AdamW (SE bloques) | dev paso 400 / 800 / 1200 / 2600 | CPU-s de entrenamiento |
|---|---|---|---|---|
| AdamW lr 3e-3 coseno (baseline) | 2.0905 | 0 | - | 6219 |
| DoG | 2.1898 | +0.099 (0.009) | 2.742 / 2.434 / 2.291 / 2.097 | 5997 |
| Prodigy | 2.3402 | +0.250 (0.012) | 2.295 / 2.226 / 2.208 / 2.314 | 6018 |
| CoinKT (W0=1e-3) | 3.6040 | +1.514 (0.026) | 3.839 / 3.776 / 3.707 / 3.589 | 6198 |
(El baseline uso 6 hilos y los demas 2 con contencion; la comparacion de tiempo es aproximada.)

## Interpretacion
- DoG sigue mejorando hasta el final (dev 2.434 -> 2.097 entre los pasos 800 y 2600): a 2600 pasos no ha saturado; el resultado depende del presupuesto. Con mas pasos podria cerrar la brecha (no medido).
- Prodigy toca su minimo de dev en el paso 1200 y luego empeora; su d se queda en 0.00699 desde el paso 400 porque el estimador r se vuelve negativo (falla de la variante simple; tambien convergio mal en el juguete: MSE 1.06 vs 0.26 de Adam).
- CoinKT no funciona con la escala inicial usada (|x| medio 42.9 -> 21.5, muy superior a pesos razonables): probable W0 mal escalada; no se probaron otros valores.

## Fallas
1) CoinKT v1: riqueza negativa y NaN (~1e34) por apostar con una escala actualizada despues de la apuesta; corregido (escala L_{t-1}, moneda recortada). 2) Prodigy: falla de metodo ya en el juguete. 3) DoG en juguete: bien (MSE 0.261 vs 0.258 de Adam). 4) No se corrieron semilla 1 ni sensibilidad x3/÷3 de la constante inicial (cada corrida ~58 min de pared bajo contencion).

## Constantes libres (cuenta honesta)
DoG: 2 (reps=1e-6, momento 0.9; sin LR, wd, clip ni schedule). Prodigy: d0=1e-6 inicial + betas (0.9, 0.999) + eps 1e-8 + lr=1 (convencion). CoinKT: 2 (W0=1e-3, momento 0.9), W0 sin calibrar. Comunes: 2600 pasos, batch 64, ctx 128, semilla 0.

## Proximos pasos sugeridos (agente)
DoG con >=5000 pasos; probar W0 de CoinKT en varios ordenes de magnitud con dev interno; contrastar Prodigy con la implementacion oficial (`prodigyopt`) antes de concluir sobre el metodo.
Nota del orquestador: este resultado NO habilita aun un componente neuronal "sin hiperparametros"; DoG es candidato para una prueba mas larga, pero con la ganancia del ensamble medida (~0.04-0.06 bpc) el beneficio esperado de entrenar un experto neuronal sigue siendo moderado.

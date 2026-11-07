# RESULTS big2 + Ixchel v0.5 (medido; fuentes: ixchel/results/big2_eval_v05.log, pipeline_big2.log, mix15_v05.json, ref_qwen15.json, ref_tinyllama.json)

## Que es
- Corpus `big2`: 686.1M chars de entrenamiento (655.2M con repeticion de math x6, chat x4, books x6); proporciones web 57.9 / code 27.6 / math 3.5 / chat 6.8 / books 4.2 %. Se excluyeron por contenido los 752 documentos de chat de validacion (corrige la contaminacion de big1).
- Modelo: Ixchel v0.5 (v0.4 + atomos de codigo podados: CK, CB, CI, CF, CS, CSK). 1,172,685,336 celdas (16.9 GB), 18 expertos mezclados con PoE global (Newton, sin tasa de aprendizaje).
- Construccion fuera de memoria en 2 tandas de 2 procesos: 02:17 -> 03:46 (89 min de pared), PoE + eval hasta 03:57. Ajuste del PoE: 109.5 s.

## bpc por dominio (val, 30k posiciones/dominio; dev PoE = 1.4469)
| dominio | big1 (v0.4) | big2 (v0.5) | delta | Qwen2.5-1.5B | TinyLlama-1.1B |
|---|---|---|---|---|---|
| web | 1.593 | 1.595 | +0.002 | 0.925 | 0.905 |
| code | 1.355 | 1.309 | -0.046 | 0.692 | 0.798 |
| math | 1.468 | 1.413 | -0.055 | 0.659 | 0.815 |
| chat | invalido (contaminado) | **1.429 (limpio)** | - | 0.696 | 0.757 |
| books | 2.062 | 2.053 | -0.009 | 1.294 | 1.380 |
Notas: el delta big1->big2 mezcla DOS cambios (v0.5 y el corpus rebalanceado); no se puede atribuir por separado. Los modelos de referencia se midieron sobre 24 segmentos x 1500 chars del val de mix1, no sobre las mismas 30k posiciones; la comparacion es de orden de magnitud, no pareada.

## Opcion multiple y programacion
| prueba | Ixchel big2 | azar | Qwen2.5-1.5B | TinyLlama-1.1B |
|---|---|---|---|---|
| HellaSwag acc_norm (n=5000 / 500) | 28.8 % (SE 0.6) | 25 | 67.2 | 58.4 |
| PIQA acc_norm (n=1838 / 400) | 56.5 % (SE 1.2) | 50 | 77.5 | 73.0 |
| ARC-Easy acc_norm (n=2376 / 400) | 30.1 % (SE 0.9) | 25 | 70.5 | 57.3 |
| HumanEval pass@1 greedy (n=164) | **0 / 164** | - | 37.2 % | 0.6 % |
Sintaxis valida de las generaciones de codigo: 15 % frente a 27.5 % de la continuacion real (n=120). De las completaciones de HumanEval, solo 18.9 % no estaban en blanco y ninguna era sintacticamente valida.

## Velocidad
Inferencia 9,203 pos/s en el proceso de evaluacion (2.05 ms/pos), por debajo de los 16.4k de big1 por tener 18 expertos en vez de 12. Referencia: transformer de 1.1B, decodificacion 5.2 tok/s con 8 hilos fp32.

## Validacion v0.5 en mix15 (15M chars; base -> v0.5)
web 1.787 -> 1.774, code 1.549 -> 1.500, math 1.525 -> 1.488, chat 1.665 -> 1.649, books 2.105 -> 2.090; HellaSwag 27.9, PIQA 53.9, ARC-E 30.0. Dev 1.7991 -> 1.7795.

## Lectura honesta
1. v0.5 y el corpus rebalanceado mejoran codigo y matematicas en ~0.05 bpc; web no cambia; la dilucion de math/books quedo corregida.
2. Aun con 1,17 mil millones de celdas y 656M chars, Ixchel esta a ~0.6-0.7 bpc de un transformer de 1.1-1.5B en todos los dominios (equivale a ~1.7-2x los bits por caracter).
3. Las pruebas que miden razonamiento o capacidad (HellaSwag, PIQA, ARC, HumanEval) siguen casi en el azar o en 0: un modelo de conteo con contexto exacto no generaliza.
4. Conclusion de producto: este modelo no puede salir como LLM general; ver `experiments/GO_TO_MARKET.md` para los usos de nicho con hipotesis a probar.

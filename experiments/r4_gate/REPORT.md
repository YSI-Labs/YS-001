# REPORT r4_gate (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/gate_r4a.json y .log)

**Conclusion: no hay ganancia neta que fusionar todavia.** La compuerta por cluster de dominio mejora codigo y matematicas pero empeora libros; la variante elegida por validacion cruzada (`alev`) resulto identica a la base (sus compuertas colapsan al vector global).

Protocolo: un solo ajuste de expertos (12 expertos, dev interno de 150k posiciones, tablas sobre train[:cut], corpus mix15 de 15M caracteres); todas las mezclas comparten sus salidas. Seleccion solo con CV en dev (2 pliegues por bloques de 500). Val (5 x 30k posiciones, mismas que mix15_base) solo para reportar. Control: la variante `global` reproduce la base exactamente (delta 0.0).

## bpc por dominio, pareado contra la base (delta = variante - base; SE por bloques de 500)
| variante | web | code | math | chat | books | dev-CV |
|---|---|---|---|---|---|---|
| base / global | 1.7870 | 1.5488 | 1.5252 | 1.6653 | 2.1048 | 1.7995 |
| cl (8 compuertas por cluster causal) | -0.0001 (SE .0007) | **-0.0567** (SE .0054, z -10.5) | **-0.0116** (SE .0009, z -12.4) | -0.0009 (SE .0005) | **+0.0155** (SE .0010, z +15.3) | 1.7958 |
| alev (10 compuertas por nivel de la cadena A; elegida por CV) | 0 | 0 | 0 | 0 | 0 | 1.7911 |
| cl_alev (80 compuertas) | 0 | 0 | 0 | 0 | 0 | 1.7995 |
bpc de `cl` en val: web 1.7869, code 1.4921, math 1.5136, chat 1.6644, books 2.1203.
Opcion multiple (variante elegida `alev` = base): HellaSwag 27.9% (azar 25), PIQA 53.0% (azar 50), ARC-E 31.0% (azar 25); coinciden exactamente con mix15_base.

## Hallazgos
1. `alev` NO es valida: en el ajuste completo tau^2 = 0, asi que sus pesos son el vector global; la mejora en CV (1.7911) sugiere tau^2 > 0 en los pliegues, pero no se guardo tau^2 por pliegue. El estimador de momentos basado en tr(C^-1) de compuertas pequenas puede dar tau^2 negativo: fallo de estabilidad del encogimiento.
2. `cl`: mejora dev-CV (-0.0037) y codigo/matematicas en val, pero EMPEORA libros: el dev se pondera por la mezcla del train (donde libros pesa poco) y val pesa los cinco dominios por igual; los pesos de compuerta se ajustan a la mezcla dominante. El efecto es real (z > 2) pero mixto: promedio de los cinco dominios -0.011 bpc.
3. Detector de dominio: el BIC sigue bajando en el tope k=8 y el reparto es muy desigual (un cluster con 88,638 de 150,000 posiciones; dos con menos de 200).
4. Coherencia con la hipotesis: el cache largo y la compuerta diferencian dominios (codigo/matematicas ganan), pero la compuerta mezcla regimenes que en val pesan distinto que en dev.

## Siguientes pasos sugeridos (agente)
Guardar tau^2 por pliegue y fijar un minimo de muestras por compuerta; ponderar el dev por dominio (equiponderado); repetir `cl` con esa ponderacion y fusionar solo si mejora libros sin perder codigo/matematicas.
Codigo: experiments/r4_gate/gatepoe.py (fit_gated, mix_logp, CausalDomainGate), run_gate.py. No se toco ixchel/core_v04.py.

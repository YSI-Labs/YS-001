# REPORT r2_lexstruct (transcrito por el orquestador desde el hand-back del agente Haiku; cifras de results/lex_*.json)
**Veredicto:** los contextos SALTADOS que conservan c1 aportan; hablante/estructura de verso casi nada. Combo recomendado: -0.030 bpc sobre v0.1 (2.0784 -> 2.0475). Se midio con la mezcla lineal EM de v0.1 (no con la geometrica de v0.2).

Atomos nuevos (28..35): spk, spos, plend, pl1f, pls, lsyl, wcs, pwcs (los 28 atomos v0.1 verificados identicos). Cabecera de hablante: linea solo con A-Z, espacio, `. ' -`, termina en `.`, largo<=40 (89 hablantes, asignado al 75% de posiciones).

## Ablaciones (cada cadena anadida a A+W+S+L; val n_eval=20000; leak_free=True)
| variante | bpc val | delta | dev mezcla | params | pico MB |
|---|---|---|---|---|---|
| base v0.1 | 2.0784 | 0 | 1.9878 | 15.65M | 2080 |
| Pw (hablante,wp) | 2.0786 | +0.0002 | 1.9877 | 17.16M | 2145 |
| Pc (hablante,c1..c3) | 2.0786 | +0.0002 | 1.9877 | 18.19M | 2153 |
| Ps (hablante,spos) | 2.0787 | +0.0003 | 1.9876 | 17.67M | 2182 |
| Ve (final linea previa) | 2.0788 | +0.0005 | 1.9869 | 17.03M | 2136 |
| Vf (1er char linea previa) | 2.0781 | -0.0003 | 1.9847 | 16.37M | 2128 |
| Vm (silabas) | 2.0748 | -0.0036 | 1.9845 | 16.30M | 2166 |
| W2 (pw2,wp saltando pw1) | 2.0742 | -0.0042 | 1.9848 | 18.01M | 2123 |
| WL (wlen,c1,c2) | 2.0723 | -0.0061 | 1.9845 | 15.69M | 2123 |
| CS (casos de mayusc.) | 2.0767 | -0.0016 | 1.9846 | 15.84M | 2163 |
| Ka (c2,c3,c4) | 2.0795 | +0.0012 | 1.9837 | 15.85M | 2085 |
| **Kb (c1 / c1,c3 / c1..c4)** | **2.0585** | **-0.0199** | 1.9736 | 15.97M | 2087 |
| Kc (c1c2 / c1c2c4c5 / c1..c6) | 2.0676 | -0.0107 | 1.9793 | 17.39M | 2106 |
| all (11 cadenas) | 2.0490 | -0.0293 | 1.9567 | 29.35M | 3669 |
| Kb+Kc | 2.0523 | -0.0260 | 1.9680 | 17.72M | 2112 |
| Kb+Kc+WL+Vm+W2 | 2.0481 | -0.0303 | 1.9631 | 20.77M | 2471 |
| Kb+Kc+WL+Vm+W2+Vf | 2.0475 | -0.0309 | 1.9609 | 21.50M | 2639 |

## Hallazgos
- Hablante (Pw,Pc,Ps): sin aporte (cambia poco dentro de una escena; (hablante,c1..c3) ~ cadena A). Ve empeora (+0.0005). Ka (c2..c4 sin c1) empeora: los contextos dispersos SIN el caracter inmediato pierden senal.
- Los saltos que conservan c1 (Kb, Kc) son los mejores: "c1,c3" captura patrones de silaba/palabra corta.
- No aditivo: "all" (11 cadenas) es peor en val que el combo de 5-6 (2.0490 vs 2.0475): mas cadenas diluyen la mezcla.
- Memoria: "all" pico 3.67 GB; combo de 6: 2.64 GB. Atomos nuevos cuestan columnas int64 de A (~1.1 GB para 33 columnas en 4.8M posiciones): bajar a int32.
- SESGO DE SELECCION (declarado por el agente): vio val del lote de cadenas individuales antes de elegir los combos (la eleccion se guio por dev, pero val estuvo a la vista).

## Hiperparametros nuevos (a mano, a priori)
tope de `spos`=3; largo maximo de cabecera=40; topes de silabas=15; vocal=aeiou; forma de cada cadena (11 candidatas = busqueda de estructura); sin learning rates.

## Parche (el orquestador solo porta las cadenas que usan atomos ya existentes: Kb, Kc, WL, W2)
Kb=[[],[c1],[c1,c3],[c1,c2,c3,c4]]; Kc=[[],[c1,c2],[c1,c2,c4,c5],[c1..c6]]; WL=[[],[wlen],[wlen,c1,c2]]; W2=[[],[wp],[wp,pw2]]. (Vm requiere atomos pls/lsyl: omitida, aporta -0.004.)
Nota del orquestador: el agente compara con "GRU 2.314"; la GRU de 30 min da 2.070 (ver r1_baseline_torch_long).

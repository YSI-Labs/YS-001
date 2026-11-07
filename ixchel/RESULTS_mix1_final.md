# Ixchel v0.4 en mix1 - corrida final (85.6M caracteres de entrenamiento) - ixchel/results/mixfinal_v04full.json y .log

Entrenamiento: 200,347,198 pares de conteo, 1268 s de pared (1388 CPU-s), RAM pico 16.4 GB; dev PoE 1.556 bpc. Pesos PoE: A 0.50, W 0.26, S 0.12, L 0.03, Kb -0.09, Kc -0.03, WL -0.27, W2 0.01, C8_W256 0.04, C8_W1024 0.08, C8_W4096 0.14, C8_Winf 0.12

## bpc por dominio (30,000 posiciones por dominio) y precision top-1 por caracter; comparacion con GRU 0.84M AdamW (4000 pasos = 33M tokens, 0.4 epocas) y con el ensamble Ixchel x GRU
| dominio | Ixchel v0.4 bpc | top-1 | GRU bpc | ensamble Ixchel+GRU bpc |
|---|---|---|---|---|
| web | 1.693 | 65.8% | 2.083 | 1.683 |
| code | 1.405 | 74.8% | 2.173 | 1.396 |
| math | 1.415 | 71.8% | 1.915 | 1.399 |
| chat | 1.522 | 69.8% | 1.999 | 1.521 |
| books | 1.995 | 58.4% | 2.370 | 1.976 |

SE por bloques de los bpc: 0.01-0.025. Pesos del ensamble (dev alineado): Ixchel 0.857, GRU 0.187; dev: Ixchel 1.565, GRU 2.029.

## Opcion multiple (acc_norm; azar entre parentesis)
| prueba | Ixchel v0.4 | GRU 0.84M |
|---|---|---|
| hellaswag (azar 25%) | 28.2% +-0.6 (n=5000) | 25.3% +-0.6 |
| piqa (azar 50%) | 54.7% +-1.2 (n=1838) | 51.1% +-1.2 |
| arc_easy (azar 25%) | 29.4% +-0.9 (n=2376) | 26.3% +-0.9 |

## Codigo
- HumanEval pass@1 voraz: **0/164 = 0%**. Completaciones con >=10 caracteres no blancos: 17.1%; sintacticamente validas entre esas: 0.6% (la cifra 'syntax_valid_frac'=83.5% es engañosa: cuenta completaciones de solo espacios).
- Continuaciones sobre codigo NO visto (120 prefijos, T=0.8): validez sintactica generada 15.8% vs continuacion real 27.5% (criterio estricto con fragmentos).
- GRU: HumanEval no se completo (proceso detenido para liberar CPU); se espera 0%.

## Notas
- La bateria del ensamble (opcion multiple y codigo) se interrumpio para liberar CPU para la construccion grande; solo se registran sus bpc por dominio (ganancia 0.001-0.02 bpc sobre Ixchel solo).
- Muestras a T=1.0 (ver JSON): siguen degenerando en repeticiones por el cache largo sobre texto propio.

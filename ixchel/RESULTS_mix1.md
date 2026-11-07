# Ixchel v0.3 en mix1 (linea base; 70M de 85.6M caracteres de entrenamiento) - ixchel/results/mix_base70M.json

Entrenamiento: 172,423,374 pares de conteo, 1991 s de pared (1923 CPU-s, maquina compartida), RAM pico 13.1 GB; dev (5% final del train) PoE = 1.619 bpc.

## bpc por dominio (30,000 posiciones por dominio; documentos retenidos; codigo = paquetes scipy+networkx NO vistos)
| dominio | bpc | SE iid | SE bloques | val chars |
|---|---|---|---|---|
| web | 1.726 | 0.013 | 0.024 | 1,007,363 |
| code | 1.496 | 0.013 | 0.024 | 1,023,551 |
| math | 1.445 | 0.012 | 0.011 | 730,400 |
| chat | 1.560 | 0.012 | 0.019 | 634,078 |
| books | 2.009 | 0.013 | 0.020 | 536,038 |

## Opcion multiple por log-verosimilitud (acc normalizada por caracteres; azar entre parentesis)
- hellaswag: acc 27.4% / acc_norm 28.9% (SE 0.6 pts, n=5000; azar 25%)
- piqa: acc 55.9% / acc_norm 54.4% (SE 1.2 pts, n=1838; azar 50%)
- arc_easy: acc 27.9% / acc_norm 29.4% (SE 0.9 pts, n=2376; azar 25%)

## Codigo
- HumanEval pass@1 (decodificacion voraz, 164 problemas, ejecucion aislada): **0.0%** (0/164).
- ATENCION: `syntax_valid_frac`=0.957 en esta corrida esta INFLADA: las completaciones son casi todo espacios en blanco (ver muestras), que son sintacticamente validas. La metrica se corrigio despues (nonblank_frac, syntax_valid_among_nonblank) y se reporta en la corrida final.
- Validez sintactica de continuaciones sobre codigo no visto (120 prefijos, T=0.8): generado 11.7% vs continuacion real 27.5% con el mismo criterio estricto.

## Muestras (T=0.8, ventana de cache 1500 chars) - degeneran en BUCLES de repeticion: el cache copia el propio texto generado
**web**
```
 he history of the cells hysteria#faity begins he her dairy caitif the history of the espressostetry beheldr gianorgo the sofa! canonicalize=hidden_states hidden_states history of the nervous soy  ous begins history of the city of the herds the history of the sofa! the nervous system.  The city cent
```
**code**
```
    attribute name
        attribute name
                                   attribute name
                                                                       """
    attribute name
        aten::cllbnat,er.B.I  attribute name
        attribute name
                     attribute name
          
```
**math**
```
 A and destit]ihog<c, liters of milk producers of per day.  Each cow farming worse. How many producers of producers of liters of milk per day.  Each ceramihs of miniature worse. ohn  fd )see er un more serious burs forming Ensemble module were serious worse. or termina has arrive. one for terminated
```
**chat**
```
Gil Response:
Give me three tips to irresponsible three tips to irresponsible three tips to articular to learn al l!tar articular to learn all articular to three tips to it.
Gocurssi rg articular Response:
Tarpon iconseee ahree tips to continue to artificial learn all  to learn to three tips to arti
```
**books**
```
or at the tata at the timesteps at the time to be, or catalog  or at the timesteps at the to dustor catalog 
no m?r t.t, guidance to deposit. s at the time to be to be, or at the time to me to be at the time to be at the time to be at the time to me to depository immed steps bewrayanor imhe time to 
```

Lectura: el modelo aprende forma local (bpc por debajo de 2 en todos los dominios salvo libros) pero no razona: ARC-Easy/HellaSwag/PIQA quedan 3-6 puntos sobre el azar y HumanEval=0. Las muestras revelan un defecto real de la combinacion cache largo + generacion (retroalimentacion positiva sobre texto propio); el bpc sobre texto real no lo detecta.

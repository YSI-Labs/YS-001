#!/bin/bash
# Construye el modelo grande v0.5 (big2) en 2 tandas de 2 procesos (limita el volcado a disco) y luego lo evalua.
cd "$(dirname "$0")"
echo "B2 inicio $(date)" > results/pipeline_big2.log
python -I -u run_bigbuild2.py 0 12000000 > results/bigbuild2_g0.log 2>&1 &
python -I -u run_bigbuild2.py 1 12000000 > results/bigbuild2_g1.log 2>&1 &
wait
echo "B2 tanda1 lista $(date)" >> results/pipeline_big2.log
python -I -u run_bigbuild2.py 2 12000000 > results/bigbuild2_g2.log 2>&1 &
python -I -u run_bigbuild2.py 3 12000000 > results/bigbuild2_g3.log 2>&1 &
wait
echo "B2 tanda2 lista $(date)" >> results/pipeline_big2.log
python -I -u run_big2_eval.py v05 > results/big2_eval_v05.log 2>&1
echo "B2_ALL_DONE $(date)" >> results/pipeline_big2.log

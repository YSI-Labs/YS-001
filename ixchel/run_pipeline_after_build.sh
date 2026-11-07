#!/bin/bash
# Espera a que terminen los 3 grupos de construccion y ejecuta, EN SERIE (mediciones de tiempo limpias):
# 1) costo real de redes de 25M-1.1B (8 hilos)  2) evaluacion del Ixchel grande  3) referencias reales (Qwen2.5-1.5B, TinyLlama-1.1B)
cd "$(dirname "$0")"
until [ $(grep -l BUILD_DONE results/bigbuild_g0.log results/bigbuild_g1.log results/bigbuild_g2.log 2>/dev/null | wc -l) -ge 3 ]; do sleep 20; done
echo "PIPE build listo $(date)" > results/pipeline.log
for s in 25m 125m 350m 1b; do python -I measure_train_cost.py $s 512 3 8 >> results/pipeline.log 2>&1; echo "PIPE cost $s listo $(date)" >> results/pipeline.log; done
python -I -u run_big_eval.py v04big >> results/big_eval_v04big.log 2>&1; echo "PIPE big eval listo $(date)" >> results/pipeline.log
until grep -q "OK TinyLlama" ../data/ref_models/download.log 2>/dev/null; do sleep 30; done
python -I -u eval_ref_model.py ../data/ref_models/Qwen2.5-1.5B qwen15 8 > results/ref_qwen15.log 2>&1; echo "PIPE qwen listo $(date)" >> results/pipeline.log
python -I -u eval_ref_model.py ../data/ref_models/TinyLlama-1.1B-intermediate-step-1431k-3T tinyllama 8 > results/ref_tinyllama.log 2>&1; echo "PIPE tinyllama listo $(date)" >> results/pipeline.log
echo PIPE_ALL_DONE >> results/pipeline.log

#!/usr/bin/env bash
# Barrido: rango r (pure y mix) con D=2^14, y tamano de historia D (r=256).
cd "$(dirname "$0")" || exit 1
for r in 16 64 256 1024; do
  python -I run.py --r $r --Dbits 14 --mix pure --n_eval 20000 --tag sweep_r${r}_D14_pure
  python -I run.py --r $r --Dbits 14 --mix dev  --n_eval 20000 --tag sweep_r${r}_D14_mix
done
for d in 12 16; do
  python -I run.py --r 256 --Dbits $d --mix pure --n_eval 20000 --tag sweep_r256_D${d}_pure
  python -I run.py --r 256 --Dbits $d --mix dev  --n_eval 20000 --tag sweep_r256_D${d}_mix
done
echo DONE

cd "$(dirname "$0")"
python -I run.py --mode random --L 32 --D 2048 --n_eval 20000 --name R0_L32_D2048 > logs/R0.log 2>&1
python -I run.py --mode ngram --K 4 --D 2048 --n_eval 20000 --name N0_K4_D2048 > logs/N0.log 2>&1
python -I run.py --mode random --L 16 --D 2048 --n_eval 20000 --name R1_L16_D2048 > logs/R1.log 2>&1
python -I run.py --mode random --L 32 --D 1024 --n_eval 20000 --name R2_L32_D1024 > logs/R2.log 2>&1
python -I run.py --mode ngram --K 4 --D 1024 --n_eval 20000 --name N1_K4_D1024 > logs/N1.log 2>&1
echo DONE > logs/sweep1.done

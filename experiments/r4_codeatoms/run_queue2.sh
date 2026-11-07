#!/bin/bash
cd "$(dirname "$0")"
python -I run_codeatoms.py prune > "results/log_prune.txt" 2>&1
python -I eval_best.py prune > "results/log_he_prune.txt" 2>&1
echo done > results/QUEUE2_DONE

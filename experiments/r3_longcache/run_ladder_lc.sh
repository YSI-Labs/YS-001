#!/bin/bash
cd "$(dirname "$0")"
for c in base wc topic dec ord6 ord8; do
  python -I run_lc.py $c 3000000 20000 >> results/ladder_lc.log 2>&1
done
echo LADDER_DONE >> results/ladder_lc.log

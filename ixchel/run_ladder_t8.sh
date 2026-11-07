#!/bin/bash
cd "$(dirname "$0")"
for spec in "full 1000000" "A 3000000" "full 3000000" "A 10000000" "full 10000000" "A 30000000" "full 30000000"; do
  python -I run_scale_text8.py $spec 50000 >> results/ladder_t8.log 2>&1
done
echo LADDER_DONE >> results/ladder_t8.log

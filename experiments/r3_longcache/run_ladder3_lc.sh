#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f results/lc_ord6wc_N3000000_n20000.json ]; do sleep 20; done
python -I run_lc.py ord8wc 3000000 20000 >> results/ladder3_lc.log 2>&1
python -I run_lc.py sbase 5000000 10000 shakespeare >> results/ladder3_lc.log 2>&1
python -I run_lc.py sord8 5000000 10000 shakespeare >> results/ladder3_lc.log 2>&1
python -I run_lc.py swc 5000000 10000 shakespeare >> results/ladder3_lc.log 2>&1
echo LADDER3_DONE >> results/ladder3_lc.log

#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f results/lc_ord6wc_N3000000_n20000.json ]; do
  if ! pgrep -f "run_lc.py ord6wc" > /dev/null 2>&1; then break; fi
  sleep 20
done
python -I run_lc.py sbase 5000000 10000 shakespeare >> results/ladder4_lc.log 2>&1
python -I run_lc.py sord8 5000000 10000 shakespeare >> results/ladder4_lc.log 2>&1
python -I run_lc.py swc 5000000 10000 shakespeare >> results/ladder4_lc.log 2>&1
echo LADDER4_DONE >> results/ladder4_lc.log

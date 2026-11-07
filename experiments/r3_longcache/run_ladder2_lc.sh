#!/bin/bash
cd "$(dirname "$0")"
while ! grep -q LADDER_DONE results/ladder_lc.log 2>/dev/null; do sleep 20; done
python -I run_lc.py ord6wc 3000000 20000 >> results/ladder2_lc.log 2>&1
python -I run_lc.py sbase 5000000 10000 shakespeare >> results/ladder2_lc.log 2>&1
python -I run_lc.py sord6 5000000 10000 shakespeare >> results/ladder2_lc.log 2>&1
python -I run_lc.py swc 5000000 10000 shakespeare >> results/ladder2_lc.log 2>&1
echo LADDER2_DONE >> results/ladder2_lc.log

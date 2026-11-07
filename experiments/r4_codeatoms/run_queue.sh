#!/bin/bash
# Cola secuencial (2 hilos, RAM<=4GB): espera al baseline mix15_base.json y luego corre full + ablacion por atomo.
cd "$(dirname "$0")"
while [ ! -f ../../ixchel/results/mix15_base.json ]; do sleep 20; done
for v in full drop:ind drop:indp drop:bdep drop:kwl drop:fwp drop:strst drop:idl drop:idp drop:endc; do
  python -I run_codeatoms.py "$v" > "results/log_${v//:/_}.txt" 2>&1
done
echo done > results/QUEUE_DONE

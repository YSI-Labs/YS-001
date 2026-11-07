for v in A AWSL; do for f in 0.06 0.125 0.25 0.5 1.0; do python -I scale_run.py $f $v; done; done
echo DONE

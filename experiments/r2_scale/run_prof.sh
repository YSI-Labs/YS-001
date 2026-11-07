for t in 1 2 4; do for mod in core core_opt; do python -I prof.py $mod $t; done; done
echo PROFDONE

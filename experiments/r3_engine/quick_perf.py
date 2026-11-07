import os, sys, time
os.environ["NUMBA_NUM_THREADS"]="1"
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(HERE)),"common"))
import harness
from engine import IxchelEngine
_,val,itos=harness.load()
e=IxchelEngine.load(os.path.join(HERE,"model_v03"))
e.generate_fast(e.new_session(val[:300]),50,seed=0)  # JIT
for L in (1000,):
    s=e.new_session(val[:L])
    t0=time.perf_counter(); txt,ids=e.generate_fast(s,5000,seed=1); t=time.perf_counter()-t0
    print("generate_fast L",L,"tok/s",5000/t,"us/tok",t/5000*1e6)
    s=e.new_session(val[:L])
    t0=time.perf_counter(); e.generate(s,2000,seed=1); t=time.perf_counter()-t0
    print("generate (python loop) tok/s",2000/t)
pos=np.arange(64,64+20000)
t0=time.perf_counter(); e.score_positions(val[:64+20000],pos); t=time.perf_counter()-t0
print("score pos/s",20000/t)

import os, sys, time
os.environ["NUMBA_NUM_THREADS"]="1"
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(HERE)),"common"))
import harness
from engine import IxchelEngine, Session, _score_stream
_,val,itos=harness.load()
e=IxchelEngine.load(os.path.join(HERE,"model_v03"))
N=20000
ids=val[:64+N]; pos=np.arange(64,64+N,dtype=np.int64)
def timed(windows, nW, label):
    kst=list(e.kstate); kst[-1]=windows
    out=np.zeros((N,e.V),np.float32)
    for rep in range(2):   # rep 0 compila; rep 1 medido
        s=Session(nW, cap=len(ids)+8, hcap=1<<12); e._grow(s,len(ids))
        t0=time.perf_counter()
        _score_stream(s.hist,s.st,s.hs,*kst,s.hkeys,s.hvals,ids,pos,e.A["isletter"],e.A["isnl"],out)
        t=time.perf_counter()-t0
    print(label,"us/pos",round(t/N*1e6,1),flush=True)
timed(np.zeros(0,np.int64),0,"solo cadenas (sin cache)")
timed(e.A["windows"],e.nW,"cadenas + cache (4 ventanas)")

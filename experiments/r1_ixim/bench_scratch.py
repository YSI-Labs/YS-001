import os
os.environ["NUMBA_NUM_THREADS"]="2"; os.environ["OMP_NUM_THREADS"]="2"; os.environ["MKL_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, time
for D in (1024, 2048):
    X = np.random.rand(8192, D).astype(np.float32)
    G = np.zeros((D, D))
    t = time.time()
    for _ in range(5):
        G += (X.T @ X).astype(np.float64)
    dt = time.time() - t
    fl = 5 * 2 * 8192 * D * D
    print(f"D={D} gram GFLOPS={fl/dt/1e9:.1f}  -> N=4.56M est {4.56e6*D*D*2/(fl/dt)/1:.0f}s")
A = np.random.rand(8192, 512).astype(np.float32); W = np.random.rand(512, 2048).astype(np.float32)
t=time.time(); 
for _ in range(5): Y = A @ W
dt=time.time()-t; print("proj GFLOPS", 5*2*8192*512*2048/dt/1e9)

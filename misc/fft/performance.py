import numpy as np
from timeit import default_timer as timer
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
from fft import gpu_fft
import pyfftw

def time_function(func, runtime=.1):
    """Time a function by running it repeatedly for at least 'runtime' seconds"""
    start = timer()
    t = 0
    count = 0

    while t < runtime:
        t0 = timer()
        func()
        tf = timer()
        t += tf - t0

        count += 1

    return t/count

nvals = 2**np.arange(5,12)

cuda_time = []
numpy_time = []
pyfftw_time = []

for n in tqdm(nvals):
    data = np.random.random(size=(n,n)).astype(np.float32)
    a = pyfftw.empty_aligned((n,n), dtype='float32')

    b = pyfftw.empty_aligned((a.shape[0], a.shape[1]//2 + 1),  dtype='complex64')
    a[...] = data

    fft_object = pyfftw.FFTW(a, b, threads=16)
    pyfftw_time.append(time_function(fft_object)*1e3)

    numpy_fft = partial(np.fft.rfft2, a=data)
    numpy_time.append(time_function(numpy_fft)*1e3)

    iterations = 10000
    cuda_fft = partial(gpu_fft, n, n, iterations)
    cuda_time.append(time_function(cuda_fft)*1e3/iterations)

numpy_time = np.array(numpy_time)
cuda_time = np.array(cuda_time)
pyfftw_time = np.array(pyfftw_time)

fig, axes = plt.subplots(ncols=2, figsize=plt.figaspect(1/2))

axes[0].loglog(nvals, numpy_time, 'o-', label='NumPy', basex=2)
axes[0].loglog(nvals, cuda_time, 'o-',  label='CUDA', basex=2)
axes[0].loglog(nvals, pyfftw_time, 'o-',  label='PyFFTW', basex=2)
axes[0].grid(ls='--', color='gray')
axes[0].set(xlabel='image size', ylabel='runtime (ms)')
axes[0].legend()

axes[1].semilogx(nvals, numpy_time/cuda_time, 'o-', basex=2, label='vs. NumPy', color='red')
axes[1].semilogx(nvals, pyfftw_time/cuda_time, 'o-', basex=2, label='vs. PyFFTW', color='k')
axes[1].legend()
axes[1].set(xlabel='image size', ylabel='CUDA speedup')
axes[1].grid(ls='--', color='gray')

plt.show()

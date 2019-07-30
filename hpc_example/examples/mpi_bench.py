import numpy as np
import dot_product as dp
from mpi4py import MPI


from IPython import get_ipython
ipython = get_ipython()

N = 1000000
x = np.random.random(N)
y = np.random.random(N)

comm = MPI.COMM_WORLD
comm.Bcast(x, root=0)
comm.Bcast(y, root=0)

ipython.magic('timeit dp.mpi_dot_product(x, y)')

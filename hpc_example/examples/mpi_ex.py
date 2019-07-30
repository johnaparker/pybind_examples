import numpy as np
import dot_product as dp
from mpi4py import MPI

comm = MPI.COMM_WORLD

N = 100
x = np.random.random(N)
y = np.random.random(N)

comm.Bcast(x, root=0)
comm.Bcast(y, root=0)

print('numpy', comm.Get_rank(), np.dot(x, y))
comm.Barrier()
print('mpi', comm.Get_rank(), dp.mpi_dot_product(x, y))

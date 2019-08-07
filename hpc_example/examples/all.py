import numpy as np
import dot_product as dp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N = 1000
x = np.random.random(N)
y = np.random.random(N)

comm.Bcast(x, root=0)
comm.Bcast(y, root=0)

if rank == 0:
    v = dp.numpy_dot_product(x, y)
    print(f'numpy: {v:.2f}')

    v = dp.dot_product(x, y)
    print(f'C++ (serial): {v:.2f}')

    v = dp.omp_dot_product(x, y)
    print(f'C++ (OpenMP): {v:.2f}')

    v = dp.gpu_dot_product(x, y)
    print(f'GPU (CUDA): {v:.2f}')

comm.Barrier()

v = dp.mpi_dot_product(x, y)
if rank == 0:
    print(f'C++ (MPI): {v:.2f}')

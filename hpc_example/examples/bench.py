import numpy as np
import dot_product as dp

from IPython import get_ipython
ipython = get_ipython()

N = 1000000
x = np.random.random(N)
y = np.random.random(N)


# print('python')
# ipython.magic('timeit dp.python_dot_product(x, y)')

print('numpy')
ipython.magic('timeit dp.numpy_dot_product(x, y)')

print('cpp')
ipython.magic('timeit dp.dot_product(x, y)')

print('cpp (openmp)')
ipython.magic('timeit dp.omp_dot_product(x, y)')

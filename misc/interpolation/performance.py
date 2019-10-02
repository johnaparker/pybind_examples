import numpy as np
from scipy.interpolate import RegularGridInterpolator
from interp import grid_interpolate
from timeit import default_timer as timer
from functools import partial

def time_function(func, runtime=0.3):
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

def f(x, y):
    return np.array([2 * x**3, 3 * y**2]*8, dtype=complex)
f = np.vectorize(f, signature='(),()->(N)')

x = np.linspace(1, 4, 100)
y = np.linspace(4, 7, 100)
data = f(*np.meshgrid(x, y, indexing='ij', sparse=True))
grid = [x,y]

f = RegularGridInterpolator((x, y), data)
X, Y = np.meshgrid(x, y, indexing='ij')

pts = np.array([X, Y])
pts = np.moveaxis(pts, 0, -1)
pts = pts.reshape([-1, 2])
pts = pts[:8]

res = f(pts)
res = grid_interpolate(grid, data, pts)

f = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=None)
time = time_function(partial(f, pts))
print(f'SciPy: {time*1e6:.2f} µs')
time = time_function(partial(grid_interpolate, grid, data, pts))
print(f'C++: {time*1e6:.2f} µs')

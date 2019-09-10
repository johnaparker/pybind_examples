import numpy as np
from scipy.interpolate import RegularGridInterpolator
from interp import grid_interpolate
from timeit import default_timer as timer
from functools import partial

def time_function(func, runtime=0.1):
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

# x = np.linspace(-1, 1, 100)
# y = np.linspace(-1, 1, 100)
# grid = [x, y]

# X, Y = np.meshgrid(x, y, indexing='ij')
# data = X**2 + Y**2


# pts = np.array([X, Y])
# pts = np.moveaxis(pts, 0, -1)
# pts = pts.reshape([-1, 2])
# pts = pts[:100]

# res = grid_interpolate(grid, data, pts)

# f = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=None)
# time = time_function(partial(f, pts))
# print(time*1e3)
# time = time_function(partial(grid_interpolate, grid, data, pts))
# print(time*1e3)


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
print(res[1])
res = grid_interpolate(grid, data, pts)
print(res[1])

f = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=None)
time = time_function(partial(f, pts))
print(time*1e3)
time = time_function(partial(grid_interpolate, grid, data, pts))
print(time*1e3)

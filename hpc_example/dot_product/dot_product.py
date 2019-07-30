import numpy as np

def python_dot_product(x, y):
    tot = 0
    for i in range(len(x)):
        tot += x[i]*y[i]

    return tot

def numpy_dot_product(x, y):
    return np.dot(x, y)

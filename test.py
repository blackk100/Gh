import numpy as np
from scipy.interpolate import interpn
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
import timeit
# import em_field as em
# import constants as const

# def value_func_3d(x, y, z):
# 	return x + y + z

# x = np.linspace(-1, 1, 10)
# y = np.linspace(-1, 1, 10)
# z = np.linspace(-1, 1, 10)
# points = (x, y, z)
# values = value_func_3d(*np.meshgrid(*points, indexing='ij'))
# xi = np.array([0, 0, 0])
# vals = interpn(points, values, xi)

# print(x)
# print(y)
# print(z)
# print(xi)
# print(values)
# print(vals)


def func1(a, b, c):
    return a**2


def func2(a, b, c):
    return np.square(a)

a = 0.99
b = np.array((5, 2, -3))
c = np.array((-10, 0, 10))
t1 = timeit.Timer(lambda: func1(a, b, c))
t2 = timeit.Timer(lambda: func2(a, b, c))
print(t1.timeit(1000000))
print(t2.timeit(1000000))
print("")

out1 = func1(a, b, c)
out2 = func2(a, b, c)
print(out1)
print(out2)
print("")
print(np.max(np.abs(out1 - out2)))
print(np.max(np.abs(out1 / out2 - 1)))
print("")

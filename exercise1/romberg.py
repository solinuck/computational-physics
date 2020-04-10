# -*- coding: utf-8 -*-

import numpy as np


def romberg(a, b, nmax, func):
    R = np.zeros((nmax, nmax), float)
    for n in range(0, nmax):
        N = 2 ** n
        R[n, 0] = trapezoidal(a, b, N, func)
        for j in range(0, n):
            R[n, j + 1] = R[n, j] + 1.0 / (4 ** (j + 1) - 1) * (R[n, j] - R[n - 1, j])
    return R[nmax - 1, nmax - 1], R


def trapezoidal(a, b, N, func):
    x, h = np.linspace(a, b, N + 1, retstep=True)
    function_values = func(x)
    if function_values[1:N].size > 0:
        s = np.sum(function_values[1:N])
    else:
        s = 0
    s = h / 2 * (function_values[0] + function_values[N - 1]) + h * s
    return s


def f1(x):
    return np.exp(x)


def f2(x):
    return np.sin(8 * x) ** 4


def f3(x):
    return np.sqrt(x)


result_1, R1 = romberg(0, 1, 5, f1)
result_2, R2 = romberg(0, 2 * np.pi, 5, f2)
result_3, R3 = romberg(0, 1, 5, f3)

print(result_1, R1)
print(result_2, R2)
print(result_3, R3)

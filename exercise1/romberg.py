# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def romberg(a, b, nmax, func):
    """
    Romberg integration.
    Arguments:
    a, b: Integration interval
    nmax: Number of slices
    func: Function to integrate
    """
    R = np.zeros((nmax, nmax), float)
    for n in range(0, nmax): # calculate terms iteratively
        N = 2 ** n
        R[n, 0] = trapezoidal(a, b, N, func)
        for j in range(0, n):
            R[n, j + 1] = R[n, j] + 1.0 / (4 ** (j + 1) - 1) * (R[n, j] - R[n - 1, j])
    return R[nmax - 1, nmax - 1], R


def trapezoidal(a, b, N, func):
    """
    trapezoidal area calculation.
    """
    x, h = np.linspace(a, b, N + 1, retstep=True)
    function_values = func(x) # generate array of sampled function values
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


analytic1 = np.exp(1) - 1
analytic2 = 3 * np.pi / 4
analytic3 = 2 / 3

n = 20

result_1, R1 = romberg(0, 1, n, f1)
result_2, R2 = romberg(0, 2 * np.pi, n, f2)
result_3, R3 = romberg(0, 1, n, f3)

# print(result_1, R1)
# print(result_2, R2)
# print(result_3, R3)

# calculate difference to analytical results
R1_ii_diff = np.abs(np.array([R1[i, i] for i in range(n)]) - analytic1)
R2_ii_diff = np.abs(np.array([R2[i, i] for i in range(n)]) - analytic2)
R3_ii_diff = np.abs(np.array([R3[i, i] for i in range(n)]) - analytic3)


def plotting(diff_array, label, ylabel="Rii - analytic", xlabel="i"):
    plt.loglog(np.arange(1, n + 1), diff_array, label=label)
    plt.xlabel("i")
    plt.ylabel(ylabel)


plotting(R1_ii_diff, label=r"$R(\int_0^1 e^{x}) - (e - 1)$")
plotting(R2_ii_diff, label=r"$R(\int_0^{2\pi} \sin^{4}{8x}) - (\frac{3 \pi}{4})$")
plotting(R3_ii_diff, label=r"$R(\int_0^1 \sqrt{x}) - \frac{2}{3}$")
plt.legend()
plt.savefig("romberg.png")

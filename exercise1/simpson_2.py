# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:04:13 2020

@author: Max
"""

import numpy as np
import matplotlib.pyplot as plt


def simpson(a, b, N, func):
    x, h = np.linspace(a, b, N, retstep=True)
    function_values = func(x)
    integral = simpson_rule(h, function_values)
    return integral, h


def simpson_rule(h, function_values):
    """Simpson rule for numerical integration.
    Arguments: function_values (numpy array)
    """

    if is_even(function_values):
        # even
        first_term = (
            h
            * (5 * function_values[-1] + 8 * function_values[-2] - function_values[-3])
            / 12
        )
        rest = simpson_rule(h, function_values[:-1:])  # recursion, bitches
        return first_term + rest
    else:
        # odd
        result = (
            h
            * (
                function_values[0]
                + function_values[-1]
                + np.sum(4 * function_values[1:-1:2])
                + np.sum(2 * function_values[2:-2:2])
            )
            / 3
        )
        return result


def is_even(array):
    """Check whether a given array is a numpy array and check if the length is even."""
    assert type(array) == np.ndarray, "Not a valid numpy array"
    if array.size % 2 == 0:
        return True
    else:
        return False


def theoretical_error(a, b, h, f):
    return (h ** 4) * f * (b - a) / 180


data = []
hs = []
es = []

a = 0
b = np.pi / 2
function = np.sin
analytic = 1

h_range = 50000
for i in range(3, h_range):
    simp_result, h = simpson(a, b, i, np.sin)
    e = theoretical_error(0, np.pi / 2, h, 1)
    data.append(np.abs(simp_result - analytic))
    hs.append(h)
    es.append(e)
    if i % 1000 == 0:
        perc = i / h_range * 100
        print(f"{perc}%")

plt.loglog(hs, data, label="Difference analytic")
plt.loglog(hs, es, label="Theoretical Error")
plt.legend()
plt.xlabel("h")
plt.ylabel("difference")
plt.savefig("simpson_2.png")

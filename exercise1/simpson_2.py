# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:04:13 2020

@author: Max
"""

import numpy as np
import matplotlib.pyplot as plt


def is_even(array):
    """Check whether a given array is a numpy array and check if the length is even."""
    assert type(array) == np.ndarray, "Not a valid numpy array"
    if array.size % 2 == 0:
        return True
    else:
        return False
    
def simpson_rule(function_values, h):
    """Simpson rule for numerical integration.
    Arguments: function_values (numpy array)
    """
    if is_even(function_values):
        #even
        first_term = h*(5*function_values[-1] + 8*function_values[-2] - function_values[-3])/12
        rest = simpson_rule(function_values[:-1:], h) # recursion, bitches
        return first_term + rest
    else:
        #odd
        result = h*(function_values[0] + function_values[-1] + np.sum(4*function_values[1:-1:2]) + np.sum(2*function_values[2:-2:2]))/3
        return(result)
        


def theoretical_error(a, b, h, f):
    return (h**4) * f * (b-a)/180

x, h = np.linspace(0, np.pi/2, 100, retstep = True)

analytic = 1
sin = np.sin(x)

data = []
hs = []
es = []
h_range = 10000
for i in range(1, h_range):
    x, h = np.linspace(0, np.pi/2, 10*i, retstep = True)
    sin = np.sin(x)
    r = simpson_rule(sin, h)
    e = theoretical_error(0, np.pi/2, h, 1)
    data.append(r-1)
    hs.append(h)
    es.append(e)
    if i % 1000 == 0:
            perc = i/h_range*100
            print(f"{perc}%")
    
plt.loglog(hs, data, label="Difference analytic")
plt.loglog(hs, es, label="Theoretical Error")
plt.legend()
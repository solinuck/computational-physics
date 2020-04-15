import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
def pendulum(time_steps, tau, x0, v0 = 0, l = 1, g = 9.81):
    x = np.zeros(time_steps)
    a0 = -g * np.sin(x0) / l
    x[0] = x0 - tau*v0 + (tau**2) * a0 / 2
    x[1] = x0


    a = np.zeros(time_steps)
    for i in range(1, time_steps-1):
        a[i] = -g * np.sin(x[i]) / l
        x[i+1] = 2 * x[i] - x[i-1] + tau**2 * a[i]

    return x

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema


def pendulum(time_steps, tau, x0, v0=0, l=1, g=9.81):
    x = np.zeros(time_steps)
    x[0] = x0

    v = np.zeros(time_steps)
    v[0] = v0

    a = np.zeros(time_steps)
    a0 = -g * np.sin(x0) / l
    a[0] = a0
    for i in range(0, time_steps - 1):
        x[i + 1] = x[i] + tau * v[i] + tau ** 2 * a[i] / 2
        a[i + 1] = -g * np.sin(x[i + 1]) / l
        v[i + 1] = v[i] + tau / 2 * (a[i] + a[i + 1])

    T = get_T(x, tau)

    return x, T


def get_T(x, tau):
    x_maxs = argrelextrema(x, np.greater)[0]
    Ts = [x[0]]
    for i in range(len(x_maxs) - 1):
        Ts.append(tau * (x_maxs[i + 1] - x_maxs[i]))

    return np.mean(Ts)


time_steps = 1000000
tau = 0.0001

x_maxs = np.linspace(0.5, np.pi - 0.1, 20)
Ts = []
for x_max in x_maxs:
    x, T = pendulum(time_steps, tau, x_max, l=10)
    plt.plot(np.linspace(0, time_steps * tau, time_steps), x)
    Ts.append(T)

plt.savefig("ideal_pendulum.png")
plt.close()

plt.plot(x_maxs, Ts)
plt.xlabel(r"$x_{max}$")
plt.ylabel("T")
plt.savefig("T_over_x_pendulum.png")
plt.close()

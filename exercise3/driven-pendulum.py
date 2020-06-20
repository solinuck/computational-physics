# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal


def runge_kutta4(N, tau, x0, v0, q):
    x = np.zeros(N)
    x[0] = x0
    v = np.zeros(N)
    v[0] = v0

    for idx, t in enumerate(
        np.linspace(0, tau * N, N)[:-1]
    ):  # runge kutta method from script
        k1 = tau * v[idx]
        k1_prim = tau * a(x[idx], v[idx], t, q)

        k2 = tau * (v[idx] + k1_prim / 2)
        k2_prim = tau * a(x[idx] + k1 / 2, v[idx] + k1_prim / 2, t + tau / 2, q)

        k3 = tau * (v[idx] + k2_prim / 2)
        k3_prim = tau * a(x[idx] + k2 / 2, v[idx] + k2_prim / 2, t + tau / 2, q)
        k4 = tau * (v[idx] + k3_prim)
        k4_prim = tau * a(x[idx] + k3, v[idx] + k3_prim, t + tau, q)
        x[idx + 1] = x[idx] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        v[idx + 1] = v[idx] + (k1_prim + 2 * k2_prim + 2 * k3_prim + k4_prim) / 6
        if x[idx + 1] > np.pi:
            x[idx + 1] -= 2 * np.pi

        if x[idx + 1] < -np.pi:
            x[idx + 1] += 2 * np.pi

    return x, v


def a(x, v, t, q, k=1, gamma=0.5, omega=2 / 3):  # acceleration from task
    return -k * np.sin(x) - gamma * v + q * np.sin(omega * t)


def plot(x, y, q, xlabel, ylabel, savepath):
    plt.plot(x, y, label="q={}".format(q))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig("{}.png".format(savepath))
    plt.close()


omega = 2 / 3
x0 = 1
v0 = 0
t0 = 2 * np.pi / omega
tau = t0 / 200

n = 8000
q1 = 0.5
x1, v1 = runge_kutta4(n, tau, x0, v0, q1)

print(x1[20], v1[20])

q2 = 0.9
x2, v2 = runge_kutta4(n, tau, x0, v0, q2)


q3 = 1.2
x3, v3 = runge_kutta4(n, tau, x0, v0, q3)


t = np.linspace(0, tau * n, n)
plot(t, x1, q1, "t", "x", "./plots/tra/{}".format(q1))
plot(t, x2, q2, "t", "x", "./plots/tra/{}".format(q2))
plot(t, x3, q3, "t", "x", "./plots/tra/{}".format(q3))

plot(x1, v1, q1, "x", "v", "./plots/phase/{}".format(q1))
plot(x2, v2, q2, "x", "v", "./plots/phase/{}".format(q2))
plot(x3, v3, q3, "x", "v", "./plots/phase/{}".format(q3))


nu_0 = omega / (np.pi * 2)

P1, Pxx_den1 = signal.periodogram(
    x1, 1 / tau, scaling="spectrum"
)  # determine power spectrum with 1/tau sampling rate for x1
plt.figure()
plt.semilogy(P1, np.sqrt(Pxx_den1), label="0.5")
plt.xlabel("Frequency")
plt.ylabel("Power")


P2, Pxx_den2 = signal.periodogram(x2, 1 / tau, scaling="spectrum")
plt.semilogy(P2, np.sqrt(Pxx_den2), label="0.9")

P3, Pxx_den3 = signal.periodogram(x3, 1 / tau, scaling="spectrum")
plt.semilogy(P3, np.sqrt(Pxx_den3), label="1.2")

plt.legend()

plt.xlim(0, 6 * nu_0)
for i in range(6):
    plt.axvline(x=nu_0 * i, alpha=0.3)
plt.savefig("plots/task3.png")
plt.show()

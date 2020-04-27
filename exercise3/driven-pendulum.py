import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift
import scipy.signal as signal

def runge_kutta4(N, tau, x0, v0, q):
    x = np.zeros(N)
    x[0] = x0
    v = np.zeros(N)
    v[0] = v0
    a = np.zeros(N)
    a[0] = acc(x0, v0, 0, q)

    for n in range(N - 1):
        k1 = tau * v[n]
        #k1_prim = tau * a[n]
        k1_prim = tau * acc(x[n], v[n], n, q)

        k2 = tau * (v[n] + k1_prim / 2)
        k2_prim = tau * acc(x[n] + k1 / 2, v[n] + k1_prim / 2, n + tau / 2, q)

        k3 = tau * (v[n] + k2_prim / 2)
        k3_prim = tau * acc(x[n] + k2 / 2, v[n] + k2_prim / 2, n + tau / 2, q)
        k4 = tau * (v[n] + k3_prim)
        k4_prim = tau * acc(x[n] + k3, v[n] + k3_prim, n + tau, q)
        x[n + 1] = x[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        v[n + 1] = v[n] + (k1_prim + 2 * k2_prim + 2 * k3_prim + k4_prim) / 6

    return x, v


def acc(x, v, t, q, k=1, gamma=0.5, omega=2 / 3):
    return -k * np.sin(x) - gamma * v + q * np.sin(omega * t)


omega = 2 / 3
q = 0.5
x0 = 1
v0 = 0
t0 = 2 * np.pi / omega
tau = t0 / 200

n = 8000
x1, v1 = runge_kutta4(n, tau, x0, v0, q)

print(x1[20], v1[20])

q = 0.9
x2, v2 = runge_kutta4(n, tau, x0, v0, q)


q = 1.2
x3, v3 = runge_kutta4(n, tau, x0, v0, q)


plt.plot(np.linspace(0, tau * n, n), x1, label="q=0.5")
plt.plot(np.linspace(0, tau * n, n), x2, label="q=0.9")
plt.plot(np.linspace(0, tau * n, n), x3, label="q=1.2")


plt.legend()
plt.savefig("trajectory.png")
plt.close()

plt.plot(x1, v1, label="q=0.5")
plt.plot(x2, v2, label="q=0.9")
plt.plot(x3, v3, label="q=1.2")
plt.savefig("phase_space.png")
plt.close()


nu_0 = omega/(np.pi*2)

P1, Pxx_den1  = signal.periodogram(x1, scaling="spectrum")
plt.plot(P1, Pxx_den1, label="q=0.5", linestyle = "dashdot", alpha=0.3, c = "grey")
P2, Pxx_den2  = signal.periodogram(x2, scaling="spectrum")
plt.plot(P2, Pxx_den2, label="q=0.9", linestyle = "dotted", alpha=0.3, c = "blue")
P3, Pxx_den3  = signal.periodogram(x3, scaling="spectrum")
plt.plot(P3, Pxx_den3, label="q=1.2", linestyle = "dashed", alpha=0.3, c = "green")


plt.xlim(0, 6*nu_0)
plt.axvline(x = nu_0, alpha = 0.3)
plt.show()

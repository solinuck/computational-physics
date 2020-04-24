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
        k1_prim = tau * a[n]
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

'''
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
'''
print(x1.size)
f1 = fftshift(x1)
print(f1.argmax()-n/2)
ff = np.abs(f1)
ff2 = ff**2


nu_0 = omega/(np.pi*2)

P,Pxx_den  = signal.periodogram(x1)
plt.plot(Pxx_den)

plt.xlim(0, 6*omega)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def harmonic_euler(time_steps, tau, x0, v0 = 0, k = 1, m = 1):
    x = np.zeros(time_steps)
    x[0] = x0
    v = np.zeros(time_steps)
    v[0] = v0
    a = np.zeros(time_steps)

    E = np.zeros(time_steps)

    for i in range(0, time_steps-1):
        a[i] = -k/m * x[i]
        v[i+1] = v[i] + tau*a[i]
        x[i+1] = x[i] + tau*v[i]
        E[i] = 0.5*m*v[i]**2  + 0.5*k*x[i]**2
    return x, E


def harmonic_verlet(time_steps, tau, x0, v0 = 0, k = 1, m = 1):
    x = np.zeros(time_steps)
    a0 = -k/m * x0
    x[0] = x0 - tau*v0 + (tau**2) * a0 / 2
    x[1] = x0
    v = np.zeros(time_steps)
    v[0] = v0

    E = np.zeros(time_steps-1)
    E[0] = 0.5 * (m * v[0]**2 + k * x[0]**2)
    a = np.zeros(time_steps)
    for i in range(1, time_steps-1):
        a[i] = -k/m * x[i]
        x[i+1] = 2 * x[i] - x[i-1] + tau**2 * a[i]
        v[i] = (x[i+1] - x[i-1]) / (2 * tau)
        E[i] =  0.5 * (m * v[i]**2 + k * x[i]**2)

    return x, E

def euler_cromer(time_steps, tau, x0, v0 = 0, k = 1, m = 1):
    x = np.zeros(time_steps)
    x[0] = x0
    v = np.zeros(time_steps)
    v[0] = v0
    a = np.zeros(time_steps)

    E = np.zeros(time_steps)

    for i in range(0, time_steps-1):
        a[i] = -k/m * x[i]
        v[i+1] = v[i] + tau*a[i]
        x[i+1] = x[i] + tau*v[i+1]
        E[i] = 0.5 * (m * v[i]**2 + k * x[i]**2)
    E[-1] = 0.5 * (m * v[-1]**2  + k * x[-1]**2)
    return x, E

x, E = euler_cromer(1000, 0.1, 10)
plt.plot(x, label="amplitude")
plt.plot(E, label="total energy")
plt.legend()
plt.show()

'''
data = harmonic_verlet(1000, 0.1, 10)
plt.plot(data)
plt.show()
'''

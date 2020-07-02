import numpy as np
from IPython import embed
import matplotlib.pyplot as plt


class NMR:
    def __init__(self, t, B, rT1, rT2, M0, phi=0):
        self.B_func = B
        self.n = t
        self.tau = 4 / self.n
        self.phi = phi
        self.t = np.arange(self.n)
        self.M = np.zeros((self.n + 1, 3))
        self.M[0] = M0

        self.C = np.diag(np.exp((-self.tau / 2) * np.array([rT2, rT2, rT1])))
        self.B = np.zeros((self.n, 3, 3))
        self.make_B()

    def make_B(self):
        B_vec = self.B_func(self.t * self.tau + self.tau / 2, self.phi)
        Omega2 = np.sum(B_vec * B_vec, axis=-1)
        Omega = np.sqrt(Omega2)
        cos = np.cos(self.tau * Omega)
        sin = np.sin(self.tau * Omega)
        self.B[:, 0, 0] = (
            B_vec[:, 0] ** 2 + (B_vec[:, 1] ** 2 + B_vec[:, 2] ** 2) * cos
        ) / Omega2
        self.B[:, 0, 1] = (
            B_vec[:, 0] * B_vec[:, 1] * (1 - cos) + Omega * B_vec[:, 2] * sin
        ) / Omega2
        self.B[:, 0, 2] = (
            B_vec[:, 0] * B_vec[:, 2] * (1 - cos) - Omega * B_vec[:, 1] * sin
        ) / Omega2

        self.B[:, 1, 0] = (
            B_vec[:, 0] * B_vec[:, 1] * (1 - cos) - Omega * B_vec[:, 2] * sin
        ) / Omega2
        self.B[:, 1, 1] = (
            B_vec[:, 1] ** 2 + (B_vec[:, 0] ** 2 + B_vec[:, 2] ** 2) * cos
        ) / Omega2
        self.B[:, 1, 2] = (
            B_vec[:, 1] * B_vec[:, 2] * (1 - cos) + Omega * B_vec[:, 0] * sin
        ) / Omega2

        self.B[:, 2, 0] = (
            B_vec[:, 0] * B_vec[:, 2] * (1 - cos) + Omega * B_vec[:, 1] * sin
        ) / Omega2
        self.B[:, 2, 1] = (
            B_vec[:, 1] * B_vec[:, 2] * (1 - cos) - Omega * B_vec[:, 0] * sin
        ) / Omega2
        self.B[:, 2, 2] = (
            B_vec[:, 2] ** 2 + (B_vec[:, 0] ** 2 + B_vec[:, 1] ** 2) * cos
        ) / Omega2

    def run(self):
        for i in self.t:
            self.M[i + 1] = self.C @ self.B[i] @ self.C @ self.M[i]


def B_field(t, phi):
    B0 = 8 * np.pi
    h = np.pi / 2
    return np.array(
        [
            h * np.cos(B0 * t + phi),
            -h * np.sin(B0 * t + phi),
            B0 * np.ones_like(t),
        ]
    ).T


f = NMR(1000, B_field, 0, 1, [1, 0, 0], phi=np.pi / 4)
f.run()

plt.grid()

plt.plot(f.M[:, 0], label=r"$M^x$")
plt.plot(f.M[:, 1], label=r"$M^y$")
plt.plot(f.M[:, 2], label=r"$M^z$")
plt.legend()
plt.show()

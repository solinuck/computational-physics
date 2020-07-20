import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


class NMR:
    def __init__(self, t, B, rT1, rT2, M0, phi=0, res=100):
        self.B_func = B
        self.n = t
        self.tau = 4 / res
        self.phi = phi
        self.t = np.arange(self.n)
        self.M = np.zeros((self.n, 3))
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
        for i in self.t[:-1]:
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


steps = 100
T1 = 0
T2 = 1
M = [1, 0, 0]
phi = np.pi / 4
f = NMR(steps, B_field, T1, T2, M, phi=phi)
f.run()


# plt.grid()
# plt.plot(f.t * f.tau, f.M[:, 0], label=r"$M^x$")
# plt.plot(f.t * f.tau, f.M[:, 1], label=r"$M^y$")
# plt.plot(f.t * f.tau, f.M[:, 2], label=r"$M^z$")
# plt.xlabel("t")
# plt.ylim(-1, 1)
# plt.title(f"Time evolution for $1/T_1 = {T1}$, $1/T_2 = {T2}$")
# plt.ylabel("Magnetization")
# plt.legend()
# plt.savefig(f"plots/M_n{steps}_{T1,T2}_M{*M,}_phi{np.round(phi, 2)}.png")
# plt.clf()

for i, m in enumerate(f.M):
    s = 10
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.quiver(0, 0, 0, m[0] / s, m[1] / s, m[2] / s)
    plt.savefig(f"anim/{i}.png")
    plt.close()

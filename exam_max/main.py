import numpy as np
from scipy.linalg import block_diag as block
import matplotlib.pyplot as plt
from tqdm import tqdm


class transmon:
    def __init__(
        self, init_state, omega, alpha, tau=0.001, T=5, time_dependent=True
    ):
        self.psi = init_state

        self.alpha = alpha * (2 * np.pi)
        self.omega = omega * (2 * np.pi)

        self.tau = tau
        self.N = int(T / tau)
        self.T = T
        self.t = tau * np.arange(self.N)
        self.time_dependent = time_dependent
        self.dim = len(init_state)

        # create ladder operators by padding a diagonal matrix
        self.a = np.pad(
            np.diag(np.sqrt(np.arange(1, self.dim))),
            [(0, 1), (1, 0)],
            mode="constant",
            constant_values=0,
        )
        N_op = self.a.T @ self.a
        # create diagonal part of Hamiltonian
        H_trans = self.omega * N_op + self.alpha / 2 * N_op @ (
            N_op - np.eye(self.dim)
        )

        self.exptrans = np.diag(np.exp(-1j * self.tau * np.diag(H_trans)))

    def Pulse(self, t):
        P = (
            np.pi
            / 2
            * np.pi
            / self.T
            * np.sin(np.pi * t / self.T)
            * np.cos(self.omega * t)
        )
        return P

    def leakage(self):
        return 1 - np.abs(self.psi[0]) ** 2 - np.abs(self.psi[1]) ** 2

    # Pauli operators
    def X(self):
        return (
            self.psi[:2].conj().T @ np.array([[0, 1], [1, 0]]) @ self.psi[:2]
        )

    def Y(self):
        return (
            self.psi[:2].conj().T
            @ np.array([[0, -1j], [1j, 0]])
            @ self.psi[:2]
        )

    def Z(self):
        return (
            self.psi[:2].conj().T @ np.array([[1, 0], [0, -1]]) @ self.psi[:2]
        )

    def update_Mat(self):
        # generate update matrix for each time step
        for t in self.t:
            evenlist = (  # all even entries of matrix
                -self.tau
                * np.sqrt(np.arange(0, self.dim, 2))
                / 2
                * self.Pulse(t)
            )
            oddlist = (  # all odd entries of matrix
                -self.tau
                * np.sqrt(np.arange(1, self.dim, 2))
                / 2
                * self.Pulse(t)
            )

            tup = tuple(  # packing it in a tupel for the constructor
                (
                    np.array(
                        [
                            [np.cos(oddlist[i]), 1j * np.sin(oddlist[i])],
                            [1j * np.sin(oddlist[i]), np.cos(oddlist[i])],
                        ]
                    )
                )
                for i in range(int((self.dim) / 2))
            )

            expK1_2 = block(*tup)  # make a block matrix

            tup = (
                tuple([1])
                + tuple(
                    (
                        np.array(
                            [
                                [
                                    np.cos(evenlist[i]),
                                    1j * np.sin(evenlist[i]),
                                ],
                                [
                                    1j * np.sin(evenlist[i]),
                                    np.cos(evenlist[i]),
                                ],
                            ]
                        )
                        for i in range(1, int((self.dim) / 2))
                    )
                )
                + tuple([1])
            )
            expK2_2 = block(*tup)

            yield (
                expK1_2 @ expK2_2 @ self.exptrans @ expK2_2 @ expK1_2
            )  # return time dependent part of Hamiltonian

    def plot_Z(self):
        plt.plot(self.t, self.Z_vals, label=r"$\langle Z\rangle$", c="b")
        plt.grid()
        plt.title(
            f"Transmon Qubit; $\\Omega /2\\pi = {self.omega/(2*np.pi):.2f}$; "
            f"{self.dim}D Case"
        )
        plt.ylim(-1.1, 1.1)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\langle Z\rangle$")
        plt.legend(loc=1)

    def plot_all(self):
        plt.plot(self.t, self.X_vals, label=r"$\langle X\rangle$", c="r")
        plt.plot(self.t, self.Y_vals, label=r"$\langle Y\rangle$", c="y")
        plt.plot(self.t, self.Z_vals, label=r"$\langle Z\rangle$", c="b")
        plt.grid()
        plt.title(
            f"Transmon Qubit; $\\Omega /2\\pi = {self.omega/(2*np.pi):.2f}$"
        )
        plt.ylim(-1.1, 1.1)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\langle A\rangle$")
        plt.legend(loc=1)

    def plot_leak(self):
        plt.plot(
            self.t,
            self.leak,
            label=f"$\\alpha/2\\pi = {self.alpha/(2*np.pi):.3f}$",
        )
        plt.xlim(0, self.T)
        plt.ylim(0)
        plt.grid()
        plt.xlabel("$t$")
        plt.ylabel("Leakage")
        plt.title("Leakage")
        plt.legend()

    def run(self):
        self.X_vals = []
        self.Y_vals = []
        self.Z_vals = []
        self.leak = []
        if self.time_dependent:
            for M in tqdm(self.update_Mat(), total=self.N):
                self.psi = M @ self.psi
                self.X_vals.append(self.X())
                self.Y_vals.append(self.Y())
                self.Z_vals.append(self.Z())
                self.leak.append(self.leakage())

        else:
            for t in tqdm(self.t):
                self.psi = self.exptrans @ self.psi
                self.X_vals.append(self.X())
                self.Y_vals.append(self.Y())
                self.Z_vals.append(self.Z())


### exercises

### free development
# init_state = [1, 1]
# init_state /= np.sqrt(2)
# a = transmon(init_state, 2, -0.1, T=5, time_dependent=False)
# a.run()
# a.plot_all()
# plt.savefig("plots1/free.png")
# plt.close()
# del a
#
# # different omegas
# for omega in tqdm([2.5, 3, 3.5, 4]):
#     a = transmon(init_state, omega, -0.1, T=5, time_dependent=False)
#     a.run()
#     a.plot_all()
#     plt.savefig(f"plots1/omega_{omega:.1f}.png")
#     plt.close()
#     del a

# with control pulse

# 2d
init_state = [1, 0]
a = transmon(init_state, 2, -0.1, T=100, time_dependent=True)
a.run()
a.plot_Z()
plt.savefig("plots1/2d_pulse.png")
plt.close()
del a

# 10d
init_state = np.zeros(10)
init_state[0] = 1
a = transmon(init_state, 2, -0.1, T=100, time_dependent=True)
a.run()
a.plot_Z()
plt.savefig("plots1/10d_pulse.png")
del a
#
# # leakage
# init_state = np.zeros(10)
# init_state[1] = 1
# alphas = [-0, -0.025, -0.050, -0.075, -0.1]
# for alpha in tqdm(alphas):
#     a = transmon(init_state, 2, alpha, T=100, time_dependent=True)
#     a.run()
#     a.plot_leak()
#     del a
# plt.savefig("plots1/leak.png")

import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import block_diag as block
import matplotlib.pyplot as plt
from tqdm import tqdm


class transmon:
    """Actually, Quantum Mechanics forbids this"""

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

        self.a = np.pad(
            np.diag(np.sqrt(np.arange(1, self.dim))),
            [(0, 1), (1, 0)],
            mode="constant",
            constant_values=0,
        )
        N_op = self.a.T @ self.a
        H_trans = self.omega * N_op + alpha / 2 * N_op @ (
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
        for t in self.t:
            evenlist = (
                -self.tau
                * np.sqrt(np.arange(0, self.dim, 2))
                / 2
                * self.Pulse(t)
            )
            oddlist = (
                -self.tau
                * np.sqrt(np.arange(1, self.dim, 2))
                / 2
                * self.Pulse(t)
            )

            tup = tuple(
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

            # expK1_2 = block_diag(
            #     [
            #         np.array(
            #             [
            #                 [np.cos(oddlist[i]), 1j * np.sin(oddlist[i])],
            #                 [1j * np.sin(oddlist[i]), np.cos(oddlist[i])],
            #             ]
            #         )
            #         for i in range(int((self.dim) / 2))
            #     ],
            # )
            expK1_2 = block(*tup)

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

            # expK2_2 = block_diag(
            #     [1]
            #     + [
            #         np.array(
            #             [
            #                 [np.cos(evenlist[i]), 1j * np.sin(evenlist[i])],
            #                 [1j * np.sin(evenlist[i]), np.cos(evenlist[i])],
            #             ]
            #         )
            #         for i in range(1, int((self.dim) / 2))
            #     ]
            #     + [1],
            # )
            yield (expK1_2 @ expK2_2 @ self.exptrans @ expK2_2 @ expK1_2)

    def run(self):
        X_vals = []
        Y_vals = []
        Z_vals = []
        if self.time_dependent:
            for M in tqdm(self.update_Mat()):
                self.psi = M @ self.psi
                X_vals.append(self.X())
                Y_vals.append(self.Y())
                Z_vals.append(self.Z())

        else:
            for t in tqdm(self.t):
                self.psi = self.exptrans @ self.psi
                X_vals.append(self.X())
                Y_vals.append(self.Y())
                Z_vals.append(self.Z())

        plt.plot(X_vals)
        plt.plot(Y_vals)
        plt.plot(Z_vals)


init_state = [1, 1]
# init_state[0] = 1
init_state /= np.sqrt(2)
a = transmon(init_state, 2, -0.1, time_dependent=True)
a.run()
plt.show()

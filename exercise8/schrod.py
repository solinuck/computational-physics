import numpy as np
from scipy.sparse import block_diag, diags
import matplotlib.pyplot as plt
from tqdm import tqdm


class schroedinger:
    def __init__(
        self, l=1001, tau=0.001, delta=0.1, m=50001, use_potential=True
    ):
        self.l = l
        self.tau = tau
        self.delta = delta
        self.m = m

        self.x = np.arange(l) * delta

        self.x0 = 20
        self.sigma = 3
        self.q = 1

        self.phi = (  # state for t = 0 as given on problem sheet
            (2 * np.pi * self.sigma ** 2) ** (-1 / 4)
            * np.exp(self.q * 1.0j * (self.x - self.x0))
            * np.exp(-((self.x - self.x0) ** 2) / (4 * self.sigma ** 2))
        )

        # matrix elements as discussed in lecture
        c = np.cos(tau / (4 * delta ** 2))
        s = np.sin(tau / (4 * delta ** 2))
        A = np.array([[c, 1.0j * s], [1.0j * s, c]])
        # use sparse matrix to store the block matrix
        K1 = block_diag(
            [A for i in range(int((self.l - 1) / 2))] + [1], format="csr"
        )
        K2 = block_diag(
            [1] + [A for i in range(int((self.l - 1) / 2))], format="csr"
        )

        # init diagonal elements of potential
        base_potential = np.ones(l) / (self.delta ** 2)
        if use_potential:
            wall = np.where(
                np.logical_and(self.x >= 50, self.x <= 50.5), 2, 0
            )  # create barrier
        else:
            wall = 0
        # create diagonal elements of potential
        potential = np.exp(-1.0j * self.tau * (base_potential + wall))
        V = diags(potential, format="csr")  # make a matrix out of it
        self.use_potential = use_potential
        self.generator = (
            K1 @ K2 @ V @ K2 @ K1
        )  # create time evolution generator

    def prob(self):  # calculate probability amplitude
        return np.abs(self.phi) ** 2

    def step(self):  # apply generator for time step once
        self.phi = self.generator @ self.phi

    def simulate(self):
        fig, ax = plt.subplots()
        plt.grid()
        # colors = ["red", "black", "yellow"]
        for i in tqdm(range(self.m)):  # fancy loading bar
            self.step()
            # print(np.max(self.prob()))
            if i in [
                0,
                5 / self.tau,
                40 / self.tau,
                45 / self.tau,
                50 / self.tau,
            ]:
                plt.xlabel("x")
                ax.set_ylabel("P(x,t)")
                if self.use_potential:
                    norm = np.sum(self.prob()[int(50.5 / self.delta) :])
                    scale = np.ones(self.l)
                    scale[int(50.5 / self.delta) :] *= norm
                    temp_phi = self.prob() * scale
                    plt.title("With potential barrier")
                    plt.axvline(50, color="grey")
                    plt.axvline(50.5, color="grey")
                    plt.plot(self.x, temp_phi, label=f"t = {i*self.tau}")
                else:
                    plt.plot(self.x, self.prob(), label=f"t = {i*self.tau}")
                    plt.title("Without potential barrier")

        plt.legend()
        pot_string = "Pot" if self.use_potential else "No_Pot"
        plt.savefig(f"static_plot/{pot_string}.png")

        plt.clf()


cat = schroedinger(use_potential=False)
cat.simulate()

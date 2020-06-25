import numpy as np
from scipy.sparse import block_diag, diags
from IPython import embed
import matplotlib.pyplot as plt
from tqdm import tqdm


class schroedinger:
    def __init__(self, l=1001, tau=0.001, delta=0.1, m=50000, use_potential=True):
        self.l = l
        self.tau = tau
        self.delta = delta
        self.m = m

        self.x = np.arange(l) * delta

        self.x0 = 20
        self.sigma = 3
        self.q = 1

        self.phi = (
            (2 * np.pi * self.sigma ** 2) ** (-1 / 4)
            * np.exp(self.q * 1.0j * (self.x - self.x0))
            * np.exp(-((self.x - self.x0) ** 2) / (4 * self.sigma ** 2))
        )

        c = np.cos(tau / (4 * delta ** 2))
        s = np.sin(tau / (4 * delta ** 2))
        A = np.array([[c, 1.0j * s], [1.0j * s, c]])
        K1 = block_diag([A for i in range(int((self.l - 1) / 2))] + [1], format="csr")
        K2 = block_diag([1] + [A for i in range(int((self.l - 1) / 2))], format="csr")

        base_potential = np.ones(l) / (self.delta ** 2)
        if use_potential:
            wall = np.where(np.logical_and(self.x >= 50, self.x <= 50.5), 2, 0)
        else:
            wall = 0
        potential = np.exp(-1.0j * self.tau * (base_potential + wall))
        V = diags(potential, format="csr")

        self.generator = K1 @ K2 @ V @ K2 @ K1

    def prob(self):
        return np.abs(self.phi) ** 2

    def step(self):
        self.phi = self.generator @ self.phi

    def simulate(self):
        for i in tqdm(range(self.m)):
            self.step()
            # print(np.max(self.prob()))
            if i % 500 == 0:
                plt.plot(self.prob())
                plt.savefig(f"plots/{i}.png")
                plt.clf()


cat = schroedinger(1001, m=100000)
cat.simulate()
embed()

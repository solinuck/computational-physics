from IPython import embed
import numpy as np
import matplotlib.pyplot as plt


class EMgrid:
    def __init__(self):
        self.lamb = 1
        self.grid = 50
        self.delta = self.lamb / self.grid
        self.tau = self.delta * 0.9
        self.length = 100
        self.lattice = self.length * self.grid + 1  # one extra bin
        self.f = 1
        self.omega = 2 * np.pi
        self.steps = 10000

        self.thickness = 2
        self.n = 1.46  # refractice index
        self.t = 0
        self.E = np.zeros(self.lattice)
        self.H = np.zeros(self.lattice - 1)  # explained on slide 15

        # physical fields at actual coordinates
        Ex = np.arange(self.lattice) * self.delta
        Hx = np.arange(self.lattice - 1) * self.delta + 0.5

        sigma = np.where(np.logical_or(Ex <= 6, self.length - 6 <= Ex), 1, 0)
        sigma_star = np.where(np.logical_or(Hx <= 6, self.length - 6 <= Hx), 1, 0)
        epsilon = np.where(
            np.logical_and(
                self.length / 2 <= Ex, Ex < self.length / 2 + self.thickness
            ),
            self.n ** 2,
            1,
        )
        self.A = (1 - sigma_star * self.tau / 2) / (1 + sigma_star * self.tau / 2)
        self.B = self.tau / (1 + sigma_star * self.tau / 2)
        self.C = (1 - sigma * self.tau / (2 * epsilon)) / (
            1 + sigma * self.tau / (2 * epsilon)
        )
        self.D = (self.tau / epsilon) / (1 + sigma * self.tau / (2 * epsilon))
        from IPython import embed

        embed()

    def source(self, t):
        return np.sin(2 * np.pi * t * self.f) * np.exp(-(((t - 30) / 20) ** 2))

    def boundary(self, x):
        if 0 <= x and x <= 6 * self.lamb:
            return 1
        elif 6 * self.lamb < x < self.lattice * self.spat_res - 6 * self.lamb:
            return 0
        elif (
            self.lattice * self.spat_res - 6 * self.lamb
            <= x
            <= self.lattice * self.spat_res
        ):
            return 1

    def plot_boundary(self):
        plt.axvspan(0, 6 * self.lamb * self.grid, alpha=0.5, color="grey")
        plt.axvspan(
            self.lattice - 6 * self.lamb * self.grid,
            self.lattice,
            alpha=0.5,
            color="grey",
        )

    @staticmethod
    def plot_material(start, stop):
        plt.axvspan(start, stop, alpha=0.5, color="green")

    def plot_E(self):
        plt.plot(self.E, color="blue")

    def plot(self):
        plt.clf()
        plt.grid()
        self.plot_boundary()
        self.plot_material(self.lattice / 2, self.lattice / 2 + self.grid)
        self.plot_E()
        plt.title(f"t = {self.t}")
        plt.ylabel("E")
        plt.xlabel("x")
        plt.xlim(0, self.lattice)
        plt.show()


sim = EMgrid()

sim.plot()
from IPython import embed

embed()

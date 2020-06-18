from IPython import embed
import numpy as np
import matplotlib.pyplot as plt


class EMgrid:
    def __init__(self):
        self.lamb = 1
        self.grid = 50
        self.spat_res = self.lamb / self.grid
        self.temp_res = self.spat_res * 0.9
        self.length = 5000
        self.f = 1
        self.omega = 2 * np.pi
        self.steps = 10000
        self.n = 1.46  # refractice index
        self.t = 0
        self.E = np.zeros(self.length)

    def source(self, t):
        return np.sin(2 * np.pi * t * self.f) * np.exp(-(((t - 30) / 20) ** 2))

    def boundary(self, x):
        if 0 <= x and x <= 6 * self.lamb:
            return 1
        elif 6 * self.lamb < x < self.length * self.spat_res - 6 * self.lamb:
            return 0
        elif (
            self.length * self.spat_res - 6 * self.lamb
            <= x
            <= self.length * self.spat_res
        ):
            return 1

    def plot_boundary(self):
        plt.axvspan(0, 6 * self.lamb * self.grid, alpha=0.5, color="grey")
        plt.axvspan(
            self.length - 6 * self.lamb * self.grid,
            self.length,
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
        self.plot_material(self.length / 2, self.length / 2 + self.grid)
        self.plot_E()
        plt.title(f"t = {self.t}")
        plt.ylabel("E")
        plt.xlabel("x")
        plt.xlim(0, self.length)
        plt.show()


sim = EMgrid()

sim.plot()

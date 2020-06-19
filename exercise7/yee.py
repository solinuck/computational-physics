import numpy as np
import matplotlib.pyplot as plt


class EMgrid:
    def __init__(self, thickness, tau_factor):
        self.lamb = 1
        self.grid = 50
        self.delta = self.lamb / self.grid
        self.tau_factor = tau_factor
        self.tau = self.delta * self.tau_factor
        self.length = 100
        self.lattice = self.length * self.grid + 1  # one extra bin
        self.f = 1
        self.omega = 2 * np.pi
        self.steps = 10000

        self.source_pos = 20 * self.grid * self.lamb
        self.thickness = thickness
        self.n = 1.46  # refractice index
        self.t = 0
        self.e = np.zeros(self.lattice)
        self.h = np.zeros(self.lattice - 1)  # explained on slide 15

        # physical fields at actual coordinates
        ex = np.arange(self.lattice) * self.delta
        hx = np.arange(self.lattice - 1) * self.delta + 0.5 * self.delta

        sigma = np.where(
            np.logical_or(ex <= 6 * self.lamb, self.length - 6 * self.lamb <= ex), 1, 0
        )
        sigma_star = np.where(
            np.logical_or(hx <= 6 * self.lamb, self.length - 6 * self.lamb <= hx), 1, 0
        )
        epsilon = np.where(
            np.logical_and(self.length / 2 <= ex, ex < self.length / 2 + self.thickness),
            self.n ** 2,
            1,
        )
        # epsilon = np.where(
        #     np.logical_or(self.length / 2 <= ex, ex < self.length / 2 + self.thickness),
        #     self.n ** 2,
        #     1,
        # )
        mu = 1
        self.a = (1 - sigma_star * self.tau / (2 * mu)) / (
            1 + sigma_star * self.tau / (2 * mu)
        )
        self.b = (self.tau / mu) / (1 + sigma_star * self.tau / (2 * mu))
        self.c = (1 - sigma * self.tau / (2 * epsilon)) / (
            1 + sigma * self.tau / (2 * epsilon)
        )
        self.d = (self.tau / epsilon) / (1 + sigma * self.tau / (2 * epsilon))

    def source(self, t):
        return np.sin(2 * np.pi * t * self.f) * np.exp(-(((t - 30) / 10) ** 2))

    def update(self):
        self.e[1 : self.lattice - 1] = (
            self.d[1 : self.lattice - 1]
            / self.delta
            * (self.h[1 : self.lattice - 1] - self.h[: self.lattice - 2])
            + self.c[1 : self.lattice - 1] * self.e[1 : self.lattice - 1]
        )
        self.e[self.source_pos] += -self.source(self.t) * self.d[self.source_pos]
        self.h = (
            self.b / self.delta * (self.e[1 : self.lattice] - self.e[: self.lattice - 1])
            + self.a * self.h
        )
        self.t += self.tau

    def simulate(self, create_anim=True, frame_delta=100):
        for n in range(self.steps):
            self.update()
            if create_anim:
                if n % frame_delta == 0:
                    self.plot(f"plots_{self.thickness}_{self.tau_factor}/{n}.png")
            if n == 4600:
                self.snap = self.e.copy()

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
        plt.plot(self.e, color="blue")

    def plot(self, fname):
        plt.clf()
        plt.grid()
        self.plot_boundary()
        self.plot_material(self.lattice / 2, self.lattice / 2 + self.grid)
        self.plot_E()
        plt.title(f"t = {self.t}, tau = {self.tau_factor}")
        plt.ylabel("E")
        plt.ylim(-0.02, 0.02)
        plt.xlabel("x")
        plt.xlim(0, self.lattice)
        plt.savefig(fname)


sim = EMgrid(2, 1.05)

sim.simulate()

reflected = np.max(sim.snap[:2500])
incident = np.max(sim.snap)

r = reflected ** 2 / incident ** 2

from IPython import embed

embed()


# sim.plot()

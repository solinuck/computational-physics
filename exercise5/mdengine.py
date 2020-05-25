import numpy as np
import itertools

import logs
import plotter


class MDEngine:
    kb = 0.83
    k_pair = 1
    k_angle = 1
    k_dihedral = 1

    def __init__(self, config, save_paths):
        self.d = config["dim"]
        self.n = config["n"]
        self.m = config["m"]
        self.l = config["l"]
        self.tau = config["tau"]
        self.lj = config["pot"]
        self.target_temp = config["target_t"]
        self.eq_e_log = logs.Logging("eq_e", save_paths["eq_e"])
        self.eq_tra_log = logs.Logging("eq_tra", save_paths["eq_tra"], console=False)

    def initialize(self):
        self.initR()
        self.initV()
        self.correctV()
        self.update(init=True)

    def initR(self):
        self.r = np.random.uniform(0, self.l, (self.n, 3))
        if self.d == 1:
            self.r[:, 1] = 0
        if self.d <= 2:
            self.r[:, 2] = 0

    def initV(self):
        temp = self.target_temp
        sig = np.sqrt((self.kb * temp) / self.m)
        self.v = np.random.normal(loc=0, scale=sig, size=(self.n, 3))
        if self.d == 1:
            self.v[:, 1] = 0
        if self.d <= 2:
            self.v[:, 2] = 0

    def correctV(self):
        if np.any(np.around(self.totMomentum(), 15)):
            self.v -= self.totMomentum() / (self.m * self.n)

    def totMomentum(self):
        return np.sum(self.m * self.v, axis=0)

    def equilibrate(self):
        self.eq_e_log.logger.info("")
        self.eq_e_log.format_log("step", "t", "temp", "ekin", "epot", "etot")

        eq_steps = 500
        for step, t in enumerate(np.linspace(0, (eq_steps - 1) * 0.01, eq_steps)):
            self.update()
            if (step % 1) == 0:
                self.thermostat(self.target_temp)
            self.eq_e_log.format_log(
                step, t, self.temp, self.ekin, self.epot, self.etot, format_nums=True
            )

            self.eq_tra_log.logger.info("")
            self.eq_tra_log.format_log(f"step = {step}", f"time = {t}")
            self.eq_tra_log.format_log("step", "x", "y", "z")
            for idx, pos in enumerate(self.r):
                self.eq_tra_log.format_log(idx, *pos)

    def update(self, init=False):
        if init:
            self.a = self.calcForce() / self.m
            self.a = np.clip(self.a, -100, 100)
        self.r = self.r + self.tau * self.v + self.tau ** 2 * self.a / 2
        self.a_nplus1 = self.calcForce() / self.m
        self.v = self.v + self.tau / 2 * (self.a + self.a_nplus1)
        self.a = self.a_nplus1.copy()
        self.ekin = np.sum(self.v ** 2 * self.m) / 2
        self.temp = 2 * self.ekin / (self.d * self.kb * (self.n - 1))
        self.etot = self.ekin + self.epot

    def thermostat(self, T):
        self.lamb = (self.d * self.kb * (self.n - 1) * T / (2 * self.ekin)) ** 0.5
        self.v = self.v * self.lamb

    def calcForce(self):
        self.f = np.zeros((self.n, 3))
        self.epot = 0
        self.A = []
        self.einheits = []
        self.directions = []
        for i1, i2 in itertools.permutations(range(self.n), 2):
            p1, p2 = self.r[i1], self.r[i2]

            dxyz = self.toroDist3D(p1, p2)

            abs = np.sum(dxyz ** 2) ** 0.5
            direction = dxyz / abs
            self.f[i1] += -direction * self.lj.force(abs)

            self.A.append([p1, self.f[i1, :]])
            self.einheits.append([p1, dxyz / abs])
            self.directions.append([p1, dxyz])
            self.epot += self.lj.pot(abs)

        self.f = np.nan_to_num(self.f)

        self.A = np.array(self.A)
        self.einheits = np.array(self.einheits)
        self.directions = np.array(self.directions)

        return self.f

    def toroDist3D(self, v1, v2):
        return np.array(
            (
                self.toroDist1D(v1[0], v2[0]),
                self.toroDist1D(v1[1], v2[1]),
                self.toroDist1D(v1[2], v2[2]),
            )
        )

    def toroDist1D(self, x1, x2):
        # get coordinates in initial box
        x1 = x1 % self.l
        x2 = x2 % self.l

        dx = x2 - x1

        if dx > (self.l / 2):
            dx = dx - self.l
        elif dx < -(self.l / 2):
            dx = dx + self.l

        return dx

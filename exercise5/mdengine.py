import numpy as np
import itertools

import logs
import plotter


class MDEngine:
    kb = 0.83
    k_pair = 1
    k_angle = 1
    k_dihedral = 1

    def __init__(self, config):
        self.d = config["dim"]
        self.n = config["n"]
        self.m = config["m"]
        self.l = config["l"]
        self.tau = config["tau"]
        self.lj = config["pot"]
        self.debug = config["debug"]
        self.target_temp = config["target_t"]
        self.step = -1

    def initialize(self):
        self.initR()
        self.initV()
        self.correctV()
        self.update(init=True)

    def initR(self):
        self.r = np.zeros((self.n, 3))
        x = np.linspace(0, self.l, int(self.n ** 0.5), endpoint=False)
        x, y = np.meshgrid(x, x)
        self.r[:, :2] = np.vstack((x.flatten(), y.flatten())).T
        self.r += self.l / self.n ** 0.5 / 2

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

    def equilibrate(self, save_paths, eq_steps=500):
        eq_e_log = logs.Logging("eq_e", save_paths["eq_e"], file=not self.debug)
        eq_tra_log = logs.Logging(
            "eq_tra", save_paths["eq_tra"], console=False, file=not self.debug
        )
        snap_log = logs.Logging(
            "snapshot", save_paths["snapshot"], console=False, file=True
        )
        eq_vel_log = logs.Logging(
            "eq_vel", save_paths["eq_vel"], console=False, file=not self.debug
        )
        eq_e_log.logger.info("")
        eq_e_log.format_log("step", "t", "temp", "ekin", "epot", "etot")

        for t in np.linspace(0, (eq_steps - 1) * 0.01, eq_steps):
            self.update()
            if (self.step % 10) == 0:
                self.thermostat(self.target_temp)

            eq_e_log.format_log(
                self.step, t, self.temp, self.ekin, self.epot, self.etot
            )

            eq_tra_log.log_r_or_v(self.step, t, self.r)
            eq_vel_log.log_r_or_v(self.step, t, self.v)
        for pos in self.r:
            snap_log.format_log(*pos)
        snap_log.format_log("@")
        for vel in self.v:
            snap_log.format_log(*vel)

    def update(self, init=False):
        if init:
            self.calcForce()
            self.a = self.f / self.m
        self.r = self.r + self.tau * self.v + self.tau ** 2 * self.a / 2
        self.calcForce()
        self.a_nplus1 = self.f / self.m
        self.v = self.v + self.tau / 2 * (self.a + self.a_nplus1)
        self.a = self.a_nplus1.copy()

        self.ekin = np.sum(self.v ** 2 * self.m) / 2
        self.temp = 2 * self.ekin / (self.d * self.kb * (self.n - 1))
        self.etot = self.ekin + self.epot
        self.step += 1

    def thermostat(self, t):
        self.lamb = (self.d * self.kb * (self.n - 1) * t / (2 * self.ekin)) ** 0.5
        self.v = self.v * self.lamb

    def calcForce(self):
        self.f = np.zeros((self.n, 3))
        self.epot = 0
        for i1, i2 in itertools.permutations(range(self.n), 2):
            p1, p2 = self.r[i1], self.r[i2]

            dxyz = self.toroDist3D(p1, p2)

            abs = np.sum(dxyz ** 2) ** 0.5
            direction = dxyz / abs
            self.f[i1] += -direction * self.lj.force(abs)

            # plotter.plot_box(self.r, self.v, self.l, self.step)  # velocities
            # plotter.plot_box(self.r, dxyz / abs, self.l, self.step)  # einheits
            # plotter.plot_box(self.r, dxyz, self.l, self.step)  # directions
            # plotter.plot_box(self.r, self.f, self.l, self.step)  # forces

            self.epot += self.lj.pot(abs)
        self.f = np.nan_to_num(self.f)

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

    def production(self, save_paths, prod_steps=1000):
        prod_e_log = logs.Logging("prod_e", save_paths["prod_e"], file=not self.debug)
        prod_tra_log = logs.Logging(
            "prod_tra", save_paths["prod_tra"], console=False, file=not self.debug
        )
        prod_vel_log = logs.Logging(
            "prod_vel", save_paths["prod_vel"], console=False, file=not self.debug
        )
        prod_e_log.logger.info("")
        prod_e_log.format_log("step", "t", "temp", "ekin", "epot", "etot")
        self.read_snap(save_paths["snapshot"])
        for t in np.linspace(0, (prod_steps - 1) * 0.01, prod_steps):
            self.update()

            prod_e_log.format_log(
                self.step, t, self.temp, self.ekin, self.epot, self.etot
            )

            prod_tra_log.log_r_or_v(self.step, t, self.r)
            prod_vel_log.log_r_or_v(self.step, t, self.v)

    def read_snap(self, f):
        file_object = open(f, "r")
        a, b = file_object.read().split("@")
        pos = np.fromstring(a, sep="\t").reshape(-1, 3)
        vel = np.fromstring(b, sep="\t").reshape(-1, 3)
        for i in range(len(pos)):
            self.r[i] = pos[i]
            self.v[i] = vel[i]
        self.update(init=True)
        self.step = 0

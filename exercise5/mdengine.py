import numpy as np
import itertools

import utils


class MDEngine:
    kb = 0.83

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
        """
        Initialize position in a square lattice
        So far this only works for two dimensions.
        """
        self.r = np.zeros((self.n, 3))
        x = np.linspace(0, self.l, int(self.n ** 0.5), endpoint=False)
        x, y = np.meshgrid(x, x)
        self.r[:, :2] = np.vstack((x.flatten(), y.flatten())).T
        self.r[:, :2] += self.l / self.n ** 0.5 / 2

    def initV(self):
        """
        Initialize each velocity axis according to a normal distribution
        This leads to maxwell-bolzman distribution for |v|
        """
        temp = self.target_temp
        sig = np.sqrt((self.kb * temp) / self.m)
        self.v = np.random.normal(loc=0, scale=sig, size=(self.n, 3))
        if self.d == 1:
            self.v[:, 1] = 0
        if self.d <= 2:
            self.v[:, 2] = 0

    def correctV(self):
        """
        Ensure that there is no center of mass momentum
        """
        self.v -= np.sum(self.m * self.v, axis=0) / (self.m * self.n)

    def equilibrate(self, save_paths, thermo_coupling, eq_steps=500):
        """
        Perform NVT equilibration with thermostat coupling and logging of energies,
        temperature, trajectories and velocities
        """
        eq_e_log, eq_tra_log, eq_vel_log, snap_log = utils.create_logger(
            save_paths, not self.debug
        )

        eq_e_log.format_log("step", "t", "temp", "ekin", "epot", "etot")

        for t in np.linspace(0, (eq_steps - 1) * 0.01, eq_steps):
            self.update()
            if (self.step % thermo_coupling) == 0:
                self.thermostat(self.target_temp)

            eq_e_log.format_log(
                self.step, np.round(t, 2), self.temp, self.ekin, self.epot, self.etot
            )

            eq_tra_log.log_r_or_v(self.step, t, self.r)
            eq_vel_log.log_r_or_v(self.step, t, self.v)

        """
        Write r and v into snapshot file
        """
        for pos in self.r:
            snap_log.format_log(*pos)
        snap_log.format_log("@")
        for vel in self.v:
            snap_log.format_log(*vel)

    def production(self, save_paths, prod_steps=1000):
        """
        Perform NVE production without thermostat coupling. Use snapshot from previous
        equilibration run to initialize system. Also log energies, temperatures,
        trajectories and velocities.
        """
        prod_e_log, prod_tra_log, prod_vel_log, _ = utils.create_logger(
            save_paths, not self.debug
        )
        prod_e_log.format_log("step", "t", "temp", "ekin", "epot", "etot")
        vel, pos = utils.read_v_and_r_snapshot(save_paths["snapshot"])
        self.r = np.zeros((self.n, 3))
        self.v = np.zeros((self.n, 3))
        for i in range(len(pos)):
            self.r[i] = pos[i]
            self.v[i] = vel[i]
        self.update(init=True)
        self.step = 0
        for t in np.linspace(0, (prod_steps - 1) * 0.01, prod_steps):
            self.update()

            prod_e_log.format_log(
                self.step, np.round(t, 2), self.temp, self.ekin, self.epot, self.etot
            )

            prod_tra_log.log_r_or_v(self.step, t, self.r)
            prod_vel_log.log_r_or_v(self.step, t, self.v)

    def update(self, init=False):
        """
        update r, v, and a according to velocity verlet method
        afterwards calculate energies and temperatur of the system and increase step.
        """
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
        """
        velocity rescaling as thermostat
        """
        self.lamb = (self.d * self.kb * (self.n - 1) * t / (2 * self.ekin)) ** 0.5
        self.v = self.v * self.lamb

    def calcForce(self):
        """
        Take all pairs of particles (particle1, particle2) and calculate for each of
        them the force effecting the first particle.
        First, the smallest distance between all image particles is calculated inside
        toroDist3D. The force is then calculated via the lj potential. The direction
        is the normal vector pointing from particle1 to particle2. To get the force
        which effects particle1 we need to put a minus sign in front.
        """
        self.f = np.zeros((self.n, 3))
        self.epot = 0
        for pair in itertools.permutations(range(self.n), 2):
            particle1, particle2 = pair
            coordinates1, coordinates2 = self.r[particle1], self.r[particle2]

            dxyz = self.toroDist3D(coordinates1, coordinates2)
            abs = np.sum(dxyz ** 2) ** 0.5
            direction = dxyz / abs

            self.f[particle1] += -direction * self.lj.force(abs)
            self.epot += -self.lj.pot(abs)

        self.f = np.nan_to_num(self.f)  # if r == 0, division results into nans

    def toroDist3D(self, c1, c2):
        """
        returns shortest distance for all 3 dimensions between two coordinates with
        periodic boundary condistions from the viewpoint of c1 which is sitting at
        (0, 0, 0).
                y
                ^
                |
                |
                |
                 ------- >x
        """
        return np.array(
            (
                self.toroDist1D(c1[0], c2[0]),
                self.toroDist1D(c1[1], c2[1]),
                self.toroDist1D(c1[2], c2[2]),
            )
        )

    def toroDist1D(self, x1, x2):
        """
        returns shortest distance in one dimension by considering periodic boundary
        conditions. The distance can also be negative if particles are on the other
        side.
        """
        # get coordinates in initial box
        x1 = x1 % self.l
        x2 = x2 % self.l

        # distance in the initial box
        dx = x2 - x1

        # if the distance is greater that half of the box size or smaller than
        # -half of the box side, then there exists an image particle with a smaller
        # distance
        if dx > (self.l / 2):
            dx = dx - self.l
        elif dx < -(self.l / 2):
            dx = dx + self.l

        return dx

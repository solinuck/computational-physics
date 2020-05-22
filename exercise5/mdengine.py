import numpy as np
import itertools


class MDEngine:
    kb = 1
    k_pair = 1
    k_angle = 1
    k_dihedral = 1

    def __init__(self, d, n, m, l, tau, potential):
        self.d = d
        self.n = n
        self.m = m
        self.l = l
        self.tau = tau
        self.pot = potential
        self.epot = 0
        self.ekin = 0
        self.etot = 0
        self.temp = 0

    def initialize(self):
        self.initR()
        self.initV()
        self.correctV()
        self.update(init=True)

    def initR(self):
        self.r = np.random.uniform(-self.l / 2, self.l / 2, (self.n, 3))
        if self.d == 1:
            self.r[:, 1] = 0
        if self.d <= 2:
            self.r[:, 2] = 0

    def initV(self):
        temp = 1
        sig = np.sqrt(self.m / (self.kb * temp))
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
        for t in np.linspace(0, 10 * 0.01, 10):
            self.update()
            self.storeFrame()
        print("end")

    def storeFrame(self):
        pass

    def computeEpot(self):
        self.epot = 0
        eAngle = 0
        eDihedral = 0
        for p1, p2, p3 in itertools.permutations(self.r, 3):
            eAngle += self.k_angle * self.angle_between(p1, p2, p3) ** 2
        for p1, p2, p3, p4 in itertools.permutations(self.r, 4):
            eDihedral += self.k_dihedral * (
                1 + np.cos(self.dihedral(p1, p2, p3, p4))
            )  # formula needs to be checked
        self.epot += eAngle + eDihedral

    def angle_between(self, a, b, c):
        ba = self.toroDist3D(b, a)
        bc = self.toroDist3D(c, b)

        ba_u = ba / np.linalg.norm(ba)
        bc_u = bc / np.linalg.norm(bc)

        return np.arccos(np.clip(np.dot(ba_u, bc_u), -1.0, 1.0))

    def dihedral(self, a, b, c, d):
        v1 = self.toroDist3D(b, a)
        v2 = self.toroDist3D(c, b)
        v3 = self.toroDist3D(d, c)

        v2 /= np.linalg.norm(v2)

        a = v1 - np.dot(v1, v2) * v2
        b = v3 - np.dot(v3, v2) * v2

        x = np.dot(a, b)
        y = np.dot(np.cross(v2, a), b)  # np.cross only works for 3D vector
        return np.arctan2(y, x)

    def update(self, init=False):
        self.computeEpot()
        if init:
            self.a = self.calcForce() / self.m
        self.r += self.tau * self.v + self.tau ** 2 * self.a / 2
        self.a_nplus1 = self.calcForce() / self.m
        self.v += self.tau / 2 * (self.a + self.a_nplus1)
        self.a = self.a_nplus1.copy()
        self.ekin = np.sum(self.v ** 2) * self.m / 2
        self.temp = 2 * self.ekin / (3 * self.n * self.kb)
        self.etot = self.ekin + self.epot

    def calcForce(self):
        f = np.zeros((self.n, 3))
        ePair = 0
        eNonBond = 0
        for i, (p1, p2) in enumerate(itertools.permutations(self.r, 2)):
            idx = i // (self.n - 1)

            dxyz = self.toroDist3D(p1, p2)

            f[idx, 0] += self.pot.force(dxyz[0])
            f[idx, 1] += self.pot.force(dxyz[1])
            f[idx, 2] += self.pot.force(dxyz[2])

            ePair += self.k_pair * (np.sum(dxyz ** 2)) / 2
            eNonBond += self.pot.pot(np.sum(dxyz ** 2) ** 0.5)

        f = np.nan_to_num(f)
        self.epot += ePair + eNonBond
        return f

    def toroDist3D(self, v1, v2):
        return np.array(
            (
                self.toroDist1D(v2[0], v1[0]),
                self.toroDist1D(v2[1], v1[1]),
                self.toroDist1D(v2[2], v1[2]),
            )
        )

    def toroDist1D(self, x1, x2):  # Can we somehow parallelize this?
        x1 = self.getInitialcoordinates(x1)
        x2 = self.getInitialcoordinates(x2)

        dx = x1 - x2

        if dx > (self.l / 2):
            dx = dx - self.l
        elif dx < -(self.l / 2):
            dx = dx + self.l

        return dx

    def getInitialcoordinates(self, x):
        while x > (self.l / 2):
            x -= self.l
        while x < (-self.l / 2):
            x += self.l
        return x

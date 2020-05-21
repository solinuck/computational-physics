import numpy as np
from scipy.spatial.distance import cdist, pdist
import itertools


class MDEngine:
    kb = 1
    k_pair = 1
    k_angle = 1
    k_dihedral = 1

    def __init__(self, d=2, n=5, m=1, l=2, tau=1):
        self.d = d
        self.n = n
        self.m = m
        self.l = l
        self.tau = tau
        self.epot = 0
        self.ekin = 0
        self.etot = 0
        self.temp = 0
        self.initialize()
        self.update()

    def initialize(self):
        self.initR()
        self.initV()
        self.correctV()
        self.updateFirstTime()

    def initR(self):
        self.r = np.random.uniform(-self.l / 2, self.l / 2, (self.n, self.d))

    def initV(self):
        temp = 1
        sig = np.sqrt(self.m / (self.kb * temp))
        self.v = np.random.normal(loc=0, scale=sig, size=(self.n, self.d))

    def correctV(self):
        if np.any(np.around(self.totMomentum(), 15)):
            self.v -= self.totMomentum() / (self.m * self.n)

    def totMomentum(self):
        return np.sum(self.m * self.v, axis=0)

    def updateFirstTime(self):
        a = self.calcForce() / self.m
        self.r += self.tau * self.v + self.tau ** 2 * a / 2
        self.a_nplus1 = self.calcForce() / self.m
        self.v += self.tau / 2 * (a + self.a_nplus1)
        self.a = self.a_nplus1
        self.ekin = np.sum(self.v[:, 0] ** 2 + self.v[:, 1] ** 2) * self.m / 2

    # def equalibrate(self):
    #     for t in np.linspace(0, 10 * 0.01, 10):
    #         self.update()
    #
    #     pass

    def storeFrame(self):
        pass

    def computeEpot(self):
        eAngle = 0
        eDihedral = 0
        for p1, p2, p3 in itertools.permutations(self.r, 3):
            eAngle += self.k_angle * self.angle_between(p1, p2, p3) ** 2
        for p1, p2, p3, p4 in itertools.permutations(self.r, 4):
            eDihedral += self.k_dihedral * (1 + np.cos(self.dihedral(p1, p2, p3, p4)))
        from IPython import embed; embed()
        self.epot += eAngle + eDihedral

    def angle_between(self, a, b, c):
        ba = (self.toroDist(b[0], a[0]), self.toroDist(b[1], a[1]))
        bc = (self.toroDist(c[0], b[0]), self.toroDist(c[1], b[1]))

        ba_u = ba / np.linalg.norm(ba)
        bc_u = bc / np.linalg.norm(bc)

        return np.arccos(np.clip(np.dot(ba_u, bc_u), -1.0, 1.0))

    def dihedral(self, a, b, c, d):
        v1 = (self.toroDist(b[0], a[0]), self.toroDist(b[1], a[1]))
        v2 = (self.toroDist(c[0], b[0]), self.toroDist(c[1], b[1]))
        v3 = (self.toroDist(d[0], c[0]), self.toroDist(d[1], c[1]))


        v2 /= np.linalg.norm(v2)

        a = v1 - np.dot(v1, v2)*v2
        b = v3 - np.dot(v3, v2)*v2

        x = np.dot(a, b)
        y = np.dot(np.cross(v2, a), b)  # np.cross only works for 3D vector
        from IPython import embed; embed()
        return np.arctan2(y, x)

    def update(self):
        self.computeEpot()
        self.r += self.tau * self.v + self.tau ** 2 * self.a / 2
        self.a_nplus1 = self.calcForce() / self.m
        self.v += self.tau / 2 * (self.a + self.a_nplus1)
        self.a = self.a_nplus1

        from IPython import embed; embed()

    def calcForce(self):
        f = np.zeros((self.n, self.d))
        ePair = 0
        eNonBond = 0
        for i, (p1, p2) in enumerate(itertools.permutations(self.r, 2)):
            dx = self.toroDist(p1[0], p2[0])
            dy = self.toroDist(p1[1], p2[1])

            idx = i // (self.n - 1)
            f[idx, 0] += self.ljForce(dx)
            f[idx, 1] += self.ljForce(dy)

            ePair += self.k_pair * (dx **2 + dy ** 2) / 2

            eNonBond += self.ljPot((dx ** 2 + dy ** 2) ** .5)

        from IPython import embed; embed()
        self.epot += ePair + eNonBond
        return f

    def ljForce(self, r, eps=1.65e-21, sig=3.4e-10):
        return 24 * eps / r * (2 * (sig / r) ** 12 - (sig / r) ** 6)

    def ljPot(self, r, eps=1.65e-21, sig=3.4e-10):
        return 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)

    def toroDist(self, x1, x2):
        x1 = self.getInitialcoordinates(x1)
        x2 = self.getInitialcoordinates(x2)

        dx = x1 - x2

        if dx > (self.l / 2):
            dx = dx - self.l
        elif dx < -(self.l / 2):
            dx = dx + self.l

        return dx

    def euclidean(self, dx, dy):
        return (dx ** 2 + dy ** 2) ** .5

    def getInitialcoordinates(self, x):
        while x > (self.l / 2):
            x -= self.l
        while x < (-self.l / 2):
            x += self.l
        return x

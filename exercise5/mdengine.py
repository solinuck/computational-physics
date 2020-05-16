import numpy as np
from scipy.spatial.distance import cdist, pdist
from itertools import permutations


class MDEngine:
    kb = 1

    def __init__(self, d=2, n=5, m=1, l=2, tau=1):
        self.d = d
        self.n = n
        self.m = m
        self.l = l
        self.tau = tau
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

    def update(self):
        self.r += self.tau * self.v + self.tau ** 2 * self.a / 2
        self.a_nplus1 = self.calcForce() / self.m
        self.v += self.tau / 2 * (self.a + self.a_nplus1)
        self.a = self.a_nplus1

    def calcForce(self, eps=1.65e-21, sig=3.4e-10):
        f = np.zeros((self.n, self.d))
        for i, (p1, p2) in enumerate(permutations(self.r, 2)):
            dx = self.toroidalCoord(p1[0], p2[0])
            dy = self.toroidalCoord(p1[1], p2[1])

            idx = i // (self.n - 1)
            f[idx, 0] += self.lennardJonesForce(dx)
            f[idx, 1] += self.lennardJonesForce(dy)

        from IPython import embed

        embed()

        return f

    def lennardJonesForce(self, r, eps=1, sig=1):
        return 24 * eps / r * (2 * (sig / r) ** 12 - (sig / r) ** 6)

    def toroidalCoord(self, x1, x2):
        x1 = self.getInitialcoordinates(x1)
        x2 = self.getInitialcoordinates(x2)

        dx = x1 - x2

        if dx > (self.l / 2):
            dx = dx - self.l
        elif dx < -(self.l / 2):
            dx = dx + self.l

        return dx

    def euclidean(self, dx, dy):
        return dx ** 2 + dy ** 2

    def getInitialcoordinates(self, x):
        while x > (self.l / 2):
            x -= self.l
        while x < (-self.l / 2):
            x += self.l
        return x

import numpy as np
from scipy.spatial.distance import cdist
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
        # nearestParticle = self.getNearestParticle()

        f = np.zeros((self.n, self.d))
        for i, pair in enumerate(permutations(self.r, 2)):
            dx, dy = self.toroidalCoord(pair[0], pair[1])

            idx = i // (self.n - 1)
            f[idx, 0] += self.lennardJonesForce(dx)
            f[idx, 1] += self.lennardJonesForce(dy)

        from IPython import embed

        embed()
        # f_direction = (self.r > nearestParticle).astype(int)
        # f_direction[f_direction == 0] = -1
        # f = f * f_direction

        #  array([[-3.75165770e-72, -2.44213344e-74],
        #         [ 1.18319646e-67,  6.54616566e-74],
        #         [ 3.75165770e-72,  2.44213344e-74],
        #         [ 3.72946159e-75, -2.60963080e-72],
        #         [-1.18319646e-67, -6.54616566e-74]])

        return f

    def lennardJonesForce(self, r, eps=1, sig=1):
        return 24 * eps / r * (2 * (sig / r) ** 12 - (sig / r) ** 6)

    # def getNearestParticle(self):
    #     allR = cdist(
    #         self.r, self.r, lambda a, b: self.euclidean(*self.toroidalCoord(a, b))
    #     )

    #  array([[0.        , 0.11586979, 0.7483549 , 0.45438486, 0.71028233],
    #         [0.11586979, 0.        , 1.26489742, 0.84764244, 1.19339731],
    #         [0.7483549 , 1.26489742, 0.        , 0.98308158, 0.0067171 ],
    #         [0.45438486, 0.84764244, 0.98308158, 0.        , 1.06104569],
    #         [0.71028233, 1.19339731, 0.0067171 , 1.06104569, 0.        ]])

    def toroidalCoord(self, p1, p2):
        x1 = self.getInitialcoordinates(p1[0])
        y1 = self.getInitialcoordinates(p1[1])
        x2 = self.getInitialcoordinates(p2[0])
        y2 = self.getInitialcoordinates(p2[1])

        dx = x1 - x2
        dy = y1 - y2

        if dx > (self.l / 2):
            dx = dx - self.l
        elif dx < -(self.l / 2):
            dx = dx + self.l

        if dy > (self.l / 2):
            dy = dy - self.l
        elif dy < -(self.l / 2):
            dy = dy + self.l
        return dx, dy

    def euclidean(self, dx, dy):
        return dx ** 2 + dy ** 2

    def getInitialcoordinates(self, x):
        while x > (self.l / 2):
            x -= self.l
        while x < (-self.l / 2):
            x += self.l
        return x

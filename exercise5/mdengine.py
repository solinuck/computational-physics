import numpy as np


class MDEngine:
    kb = 1

    def __init__(self, d=2, n=5, m=1, l=10, tau=1):
        self.d = d
        self.n = n
        self.m = m
        self.l = l
        self.tau = tau
        self.initR()
        self.initV()
        self.correctV()
        self.lennardJones_pot(1, 1)

    def initR(self):
        self.r = np.random.uniform(0, self.l, (self.n, self.d))

    def initV(self):
        temp = 1
        sig = np.sqrt(self.m / (self.kb * temp))
        self.v = np.random.normal(loc=0, scale=sig, size=(self.n, self.d))

    def correctV(self):
        if np.any(np.around(self.totMomentum(), 15)):
            self.v -= self.totMomentum() / (self.m * self.n)

    def totMomentum(self):
        return np.sum(self.m * self.v, axis=0)

    def vVerlet(self):
        # a = - grad(potential) / m
        a = np.ones(self.n, self.d)
        a_nplus1 = a
        self.r += self.tau * self.v + self.tau ** 2 * a / 2
        self.v += self.tau / 2 * (a + a_nplus1)

    def lennardJones_pot(self, eps, sig):
        r_ij = self.calc_distance()

        U_ij = 4 * eps * ((sig / r_ij) ** 12 - (sig / r_ij) ** 6)

        from IPython import embed

        embed()

        return U_ij

    def calc_distance(self):
        r_ij = r_ij = np.zeros((self.n, self.n))
        for idx, r_i in enumerate(self.r):
            r_ij[idx][:] = np.linalg.norm(self.r - r_i, axis=1)
        return r_ij

    # @staticmethod
    # def faculty(n):
    #     if n == 0:
    #         return 1
    #     return n * faculty(n - 1)

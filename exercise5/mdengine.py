import numpy as np


class MDEngine:
    kb = 1

    def __init__(self, d=2, n=5, m=1, l=10):
        self.d = d
        self.n = n
        self.m = m
        self.l = l
        self.init_velocity()
        self.correct_velocity()

    def init_velocity(self):
        temp = 1
        var = self.m / (self.kb * temp)
        self.v = np.random.normal(loc=0, scale=var, size=(self.n, self.d))

    def correct_velocity(self):
        if np.any(np.around(self.totMomentum(), 15)):
            self.v -= self.totMomentum() / (self.m * self.n)

    def totMomentum(self):
        return np.sum(self.m * self.v, axis=0)

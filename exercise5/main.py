from mdengine import MDEngine


class lennardJones:
    def __init__(self, eps, sig):
        self.eps = eps
        self.sig = sig

    def force(self, r):
        return 24 * self.eps / r * (2 * (self.sig / r) ** 12 - (self.sig / r) ** 6)

    def pot(self, r):
        return 4 * self.eps * ((self.sig / r) ** 12 - (self.sig / r) ** 6)


if __name__ == "__main__":
    lj = lennardJones(eps=1.65e-21, sig=3.4e-10)
    engine = MDEngine(d=2, n=5, m=1, l=2, tau=1, potential=lj)
    engine.initialize()
    engine.equilibrate()

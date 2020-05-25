import os
from pathlib import Path

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
    lj = lennardJones(eps=99.4, sig=3.4)
    config = {
        "dim": 2,
        "n": 100,
        "m": 39.9,
        "l": 37.8,
        "tau": 0.01,
        "pot": lj,
        "target_t": 150,
    }
    run = 0
    logs = Path("logs")
    eq_e = logs.joinpath("equi", "energy", f"run_{run}")
    eq_tra = logs.joinpath("equi", "tra", f"run_{run}")
    prod_e = logs.joinpath("prod", "energy", f"run_{run}")
    prod_tra = logs.joinpath("prod", "tra", f"run_{run}")

    save_paths = {
        "eq_e": eq_e,
        "eq_tra": eq_tra,
        "prod_e": prod_e,
        "prod_tra": prod_tra,
    }

    for key in save_paths:
        dir = save_paths[key].parent
        dir.mkdir(parents=True, exist_ok=True)
        run = 0
        while save_paths[key].exists():
            run += 1
            save_paths[key] = dir.joinpath("run_{}".format(run))
        save_paths[key].touch()
    engine = MDEngine(config, save_paths)
    engine.initialize()
    engine.equilibrate()

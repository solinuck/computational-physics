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
        "n": 10,
        "m": 39.9,
<<<<<<< HEAD
        "l": 37.8,  # 37.8
=======
        "l": 37.8,  # * 20,  # 37.8
>>>>>>> 4b60fd2f9452728b049c99fa3b4d965b100e835d
        "tau": 0.01,
        "pot": lj,
        "target_t": 150,
    }
    config_params = [
        f"d{config['dim']}",
        f"n{config['n']}",
        f"m{config['m']}",
        f"l{config['l']}",
        f"tau{config['tau']}",
        f"t{config['target_t']}",
    ]

    config_path = "_".join([x for x in config_params])
    save_paths = {}
    save_paths["eq_e"] = os.path.join("logs", "equi", "energy", config_path)
    save_paths["eq_tra"] = os.path.join("logs", "equi", "tra", config_path)
    save_paths["prod_e"] = os.path.join("logs", "prod", "energy", config_path)
    save_paths["prod_tra"] = os.path.join("logs", "prod", "tra", config_path)

    for save_path in save_paths.values():
        Path(save_path).mkdir(parents=True, exist_ok=True)

    engine = MDEngine(
        d=config["dim"],
        n=config["n"],
        m=config["m"],
        l=config["l"],
        tau=config["tau"],
        potential=config["pot"],
        target_temp=config["target_t"],
    )
    engine.initialize()
    engine.equilibrate(
        os.path.join(save_paths["eq_e"], "test"),
        os.path.join(save_paths["eq_tra"], "test"),
    )

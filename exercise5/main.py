import argparse
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


def create_new_files(save_paths, new_files=False):
    for key in save_paths:
        dir = save_paths[key].parent
        dir.mkdir(parents=True, exist_ok=True)
        if new_files:
            run = 0
            while save_paths[key].exists():
                run += 1
                save_paths[key] = dir.joinpath(
                    "{}.{}".format(save_paths[key].stem, run)
                )
        else:
            if save_paths[key].exists():
                save_paths[key].unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecular Dynamics Engine")
    parser.add_argument("-d", dest="debug", action="store_true", default=False)
    parser.add_argument("-n", dest="n", action="store", default=100)
    parser.add_argument(
        "--new_files", dest="new_files", action="store_true", default=False
    )
    parser.add_argument("-e", dest="eq_runs", action="store", default=1000)
    parser.add_argument("-p", dest="prod_runs", action="store", default=1000)
    parser.add_argument("--density", dest="density", action="store", default=0.07)

    args = parser.parse_args()

    lj = lennardJones(eps=99.4, sig=3.4)
    config = {
        "dim": 2,
        "n": int(args.n),
        "m": 39.9,
        "l": (int(args.n) / float(args.density)) ** 0.5,
        "tau": 0.01,
        "pot": lj,
        "target_t": 150,
        "debug": args.debug,
    }

    logs = Path("logs")
    density_dir = logs.joinpath(f"d_{args.density}")
    eq_e = density_dir.joinpath("equi", "energy", "run.0")
    eq_tra = density_dir.joinpath("equi", "tra", "run.0")
    eq_vel = density_dir.joinpath("equi", "vel", "run.0")
    prod_e = density_dir.joinpath("prod", "energy", "run.0")
    prod_tra = density_dir.joinpath("prod", "tra", "run.0")
    prod_vel = density_dir.joinpath("prod", "vel", "run.0")
    snapshot = density_dir.joinpath("snapshot", "snapshot.0")
    Path("results").mkdir(parents=True, exist_ok=True)

    save_paths = {
        "eq_e": eq_e,
        "eq_tra": eq_tra,
        "eq_vel": eq_vel,
        "prod_e": prod_e,
        "prod_tra": prod_tra,
        "prod_vel": prod_vel,
        "snapshot": snapshot,
    }
    if not args.debug:
        create_new_files(save_paths, args.new_files)
    engine = MDEngine(config)
    engine.initialize()
    engine.equilibrate(save_paths, args.eq_runs)
    engine.production(save_paths, args.prod_runs)

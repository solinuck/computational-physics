import utils
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
    """
    --eq_true for equilibration
    --new_files for not overwriting old files
    --eq_steps number of equilibration steps
    --prod_steps number of production steps
    """
    args = utils.mainParser()

    if args.eq:
        mode = "equi"
    else:
        mode = "prod"

    save_paths = utils.createPaths(mode, args.density)

    if not args.debug:
        utils.createDirsAndFiles(
            save_paths, args.density, mode, new_files=args.new_files
        )

    """
    Engine
    """

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
    engine = MDEngine(config)

    if args.eq:
        engine.initialize()
        engine.equilibrate(save_paths, thermo_coupling=10, eq_steps=int(args.eq_runs))
    else:
        engine.production(save_paths, int(args.prod_runs))

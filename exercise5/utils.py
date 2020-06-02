import argparse
from pathlib import Path


def mainParser():
    parser = argparse.ArgumentParser(description="Molecular Dynamics Engine")
    parser.add_argument("-d", dest="debug", action="store_true", default=False)
    parser.add_argument("-n", dest="n", action="store", default=100)
    parser.add_argument(
        "--new_files", dest="new_files", action="store_true", default=False
    )
    parser.add_argument("-e", dest="eq_runs", action="store", default=500)
    parser.add_argument("-p", dest="prod_runs", action="store", default=1000)
    parser.add_argument("--density", dest="density", action="store", default=0.07)
    parser.add_argument("--eq_true", dest="eq", action="store_true", default=False)

    return parser.parse_args()


def analysisParser():
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument("--density", dest="density", action="store", default=0.07)
    parser.add_argument("--window", dest="window", action="store", default=20)
    parser.add_argument("--eq_true", dest="eq", action="store_true", default=False)
    parser.add_argument("--run", dest="run_number", action="store", default=0)

    return parser.parse_args()


def createPaths(mode, density, run_number=0):
    logs = Path("logs")
    density_dir = logs.joinpath(f"d_{density}")
    e = density_dir.joinpath(mode, "energy", "run.{}".format(run_number))
    tra = density_dir.joinpath(mode, "tra", "run.{}".format(run_number))
    vel = density_dir.joinpath(mode, "vel", "run.{}".format(run_number))
    snapshot = density_dir.joinpath("snapshot", "snapshot.{}".format(run_number))

    save_paths = {"e": e, "tra": tra, "vel": vel, "snapshot": snapshot}
    return save_paths


def createDirsAndFiles(save_paths, new_files=False):
    Path("results").mkdir(parents=True, exist_ok=True)
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

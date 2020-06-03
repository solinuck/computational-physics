import argparse
from pathlib import Path
import numpy as np

import logs


"""
######################## Parser ########################
"""


def mainParser():
    parser = argparse.ArgumentParser(description="Molecular Dynamics Engine")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    parser.add_argument("--n_particles", dest="n", action="store", default=100)
    parser.add_argument(
        "--new_files", dest="new_files", action="store_true", default=False
    )
    parser.add_argument("--eq_runs", dest="eq_runs", action="store", default=1000)
    parser.add_argument("--prod_runs", dest="prod_runs", action="store", default=1000)
    parser.add_argument("--density", dest="density", action="store", default=0.07)
    parser.add_argument("--eq_true", dest="eq", action="store_true", default=False)

    return parser.parse_args()


def analysisParser():
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument("--density", dest="density", action="store", default=0.07)
    parser.add_argument("--window", dest="window", action="store", default=20)
    parser.add_argument("--eq_true", dest="eq", action="store_true", default=False)
    parser.add_argument("--run", dest="run", action="store", default=0)

    return parser.parse_args()


"""
######################## Working with paths and logs ########################
"""


def createPaths(mode, density, run_number=0):
    logs = Path("logs")
    density_dir = logs.joinpath(f"d_{density}")
    e = density_dir.joinpath(mode, "energy", "run.{}".format(run_number))
    tra = density_dir.joinpath(mode, "tra", "run.{}".format(run_number))
    vel = density_dir.joinpath(mode, "vel", "run.{}".format(run_number))
    snapshot = density_dir.joinpath("snapshot", "snapshot.{}".format(run_number))

    save_paths = {"e": e, "tra": tra, "vel": vel, "snapshot": snapshot}
    return save_paths


def createDirsAndFiles(save_paths, density, mode, new_files=False):
    Path("results/d_{}".format(density)).mkdir(parents=True, exist_ok=True)
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
            if save_paths[key].exists() and mode == "equi":
                save_paths[key].unlink()
            if save_paths[key].exists() and mode == "prod":
                if key != "snapshot":
                    save_paths[key].unlink()


def create_logger(save_paths, write_file):
    eq_e_log = logs.Logging("e", save_paths["e"], file=write_file)
    eq_tra_log = logs.Logging("tra", save_paths["tra"], console=False, file=write_file)
    eq_vel_log = logs.Logging("vel", save_paths["vel"], console=False, file=write_file)
    snap_log = logs.Logging(
        "snapshot", save_paths["snapshot"], console=False, file=write_file
    )
    return eq_e_log, eq_tra_log, eq_vel_log, snap_log


"""
######################## Reading data ########################
"""


def read_e_and_t(fname):
    e = []
    temp = []
    with open(fname) as f:
        next(f)
        for line in f:
            temp.append(np.fromstring(line, sep="\t")[2])
            e.append(np.fromstring(line, sep="\t")[5])
    return temp, e


def read_v_and_r_snapshot(fname):
    file_object = open(fname, "r")
    a, b = file_object.read().split("@")
    pos = np.fromstring(a, sep="\t").reshape(-1, 3)
    vel = np.fromstring(b, sep="\t").reshape(-1, 3)
    return vel, pos


def read_v_or_r(fname):
    with open(fname) as f:
        frame = ""
        for line in f:
            if not str.isdigit(line[0]):  # skip lines without number
                continue
            frame += line
        values = (
            np.fromstring(frame, sep="\t").reshape(-1, 4)[:, 1:4].reshape(-1, 100, 3)
        )
    return values

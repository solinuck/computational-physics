from pathlib import Path
import argparse

import numpy as np

import plotter

parser = argparse.ArgumentParser(description="Analysis")
parser.add_argument("--density", dest="density", action="store", default=0.07)
parser.add_argument("--window", dest="window", action="store", default=20)
args = parser.parse_args()


logs = Path("logs")
density_dir = logs.joinpath(f"d_{args.density}")
eq_e = density_dir.joinpath("equi", "energy", "run.0")
eq_tra = density_dir.joinpath("equi", "tra", "run.0")
eq_vel = density_dir.joinpath("equi", "vel", "run.0")
prod_e = density_dir.joinpath("prod", "energy", "run.0")
prod_tra = density_dir.joinpath("prod", "tra", "run.0")
prod_vel = density_dir.joinpath("prod", "vel", "run.0")
snapshot = density_dir.joinpath("snapshot", "snapshot.0")

save_paths = {
    "eq_e": eq_e,
    "eq_tra": eq_tra,
    "eq_vel": eq_vel,
    "prod_e": prod_e,
    "prod_tra": prod_tra,
    "prod_vel": prod_vel,
    "snapshot": snapshot,
}
density = density_dir.name.split("_")[1]


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


def read_v_and_r_prod(fname):
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


vs = []
for i in range(1000 // int(args.window)):
    v_prod = read_v_and_r_prod(save_paths["prod_vel"])[
        :: int(args.window)
    ]  # shape = (1000, 100, 3)
    average = np.mean(v_prod, axis=0)  # shape = (100, 3)
    result = np.sum(average ** 2, axis=1) ** 0.5  # shape = (100,)
    vs.append(result)
vs = np.array(vs).flatten()

plotter.plot_hist(result, 12)

vel, pos = read_v_and_r_snapshot(save_paths["snapshot"])
plotter.plot_box(
    pos,
    vel,
    (100 / float(args.density)) ** 0.5,
    xlabel=r"$x[\AA]$",
    ylabel=r"$y[\AA]$",
    title="position and velocities, iteration = {}".format(1000),
    savepath="results/snapshot_after_eq_d{}.png".format(density),
)


temp, e = read_e_and_t(save_paths["eq_e"])

plotter.plot_e_t(
    temp,
    tau=0.01,
    temp=True,
    title="Temperature over time, d = {}".format(density),
    ylabel=r"$T[K]$",
    xlabel=r"$t[ps]$",
    savepath="results/t_time_d{}.png".format(density),
)
plotter.plot_e_t(
    e,
    tau=0.01,
    title="Energy over time, d = {}".format(density),
    ylabel=r"$E[u \cdot \frac{\AA^2}{ps^2}]$",
    xlabel=r"$t[ps]$",
    savepath="results/e_time_d{}.png".format(density),
)

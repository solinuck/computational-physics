import numpy as np
from pathlib import Path
import argparse

import plotter
import matplotlib.animation as anim
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Analysis")
parser.add_argument("--density", dest="density", action="store", default=0.07)
parser.add_argument("--window", dest="window", action="store", default=20)
parser.add_argument("--eq", dest="eq", action="store_true", default=False)

args = parser.parse_args()

if args.eq:
    mode_str = "prod"
else:
    mode_str = "equi"

logs = Path("logs")
density_dir = logs.joinpath(f"d_{args.density}")
e = density_dir.joinpath(mode_str, "energy", "run.0")
tra = density_dir.joinpath(mode_str, "tra", "run.0")
vel = density_dir.joinpath(mode_str, "vel", "run.0")
snapshot = density_dir.joinpath("snapshot", "snapshot.0")

save_paths = {"e": e, "tra": tra, "vel": vel, "snapshot": snapshot}


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


read_r = read_v_and_r_prod(save_paths["tra"])
read_v = read_v_and_r_prod(save_paths["vel"])


fig = plt.figure()


def plot(r, vec, l):
    r_ = r % l
    x, y = r_[:, 0], r_[:, 1]
    vx, vy = vec[:, 0], vec[:, 1]
    plt.scatter(x, y)
    for i in range(r.shape[0]):
        plt.annotate(i, (x[i], y[i]))
    plt.quiver(x, y, vx, vy)
    axes = plt.gca()
    axes.set_xlim([0, l])
    axes.set_ylim([0, l])
    axes.set_aspect("equal", "box")


def animate(i):
    plt.clf()
    plot(read_r[i * 2], read_v[i * 2], (100 / float(args.density)) ** 0.5)


ani = anim.FuncAnimation(fig, animate, interval=1, frames=len(read_r))
ani.save("results/trajectory_animation_{}_d{}.mp4".format(mode_str, args.density),)

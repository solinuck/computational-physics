import numpy as np
import itertools
import matplotlib.pyplot as plt

import utils
import plotter


args = utils.analysisParser()
if args.eq:
    mode = "equi"
else:
    mode = "prod"


save_paths = utils.createPaths(mode, args.density, args.run_number)

box_width = (int(100) / float(args.density)) ** 0.5

"""
######################## Functions for reading data ########################
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


def read_v_and_r(fname):
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


"""
######################## Task 2.2 ########################
"""
temp, e = read_e_and_t(save_paths["e"])

plotter.plot_e_t(
    temp,
    tau=0.01,
    temp=True,
    title="Temperature over time, d = {}".format(args.density),
    ylabel=r"$T[K]$",
    xlabel=r"$t[ps]$",
    savepath="results/t_time_d{}.png".format(args.density),
)
plotter.plot_e_t(
    e,
    tau=0.01,
    title="Energy over time, d = {}".format(args.density),
    ylabel=r"$E[u \cdot \frac{\AA^2}{ps^2}]$",
    xlabel=r"$t[ps]$",
    savepath="results/e_time_d{}.png".format(args.density),
)

"""
######################## Task 2.3 ########################
"""

vel_snap, pos_snap = read_v_and_r_snapshot(save_paths["snapshot"])
plotter.plot_box(
    pos_snap,
    vel_snap,
    box_width,
    xlabel=r"$x[\AA]$",
    ylabel=r"$y[\AA]$",
    title="position and velocities, snapshot run: {}".format(args.run),
    savepath="results/snapshot_d{}.png".format(args.density),
)

"""
######################## Task 2.4 ########################
"""
vs = []
read_v = read_v_and_r(save_paths["vel"])
for i in range(1000 // int(args.window)):
    v_prod = read_v[: i : i + int(args.window)]  # shape = (1000, 100, 3)
    average = np.mean(v_prod, axis=0)  # shape = (100, 3)
    result = np.sum(average ** 2, axis=1) ** 0.5  # shape = (100,)
    vs.append(result)
vs = np.array(vs).flatten()

plotter.plot_hist(vs, 12)

"""
######################## Task 2.5 ########################
"""


def toroDist3D(v1, v2, l):
    return np.array(
        (
            toroDist1D(v1[0], v2[0], l),
            toroDist1D(v1[1], v2[1], l),
            toroDist1D(v1[2], v2[2], l),
        )
    )


def toroDist1D(x1, x2, l):
    # get coordinates in initial box
    x1 = x1 % l
    x2 = x2 % l

    dx = x2 - x1

    if dx > (l / 2):
        dx = dx - l
    elif dx < -(l / 2):
        dx = dx + l

    return dx


def correlation(data, bins):
    hists = []
    for i in range(data.shape[0]):
        hist, bin = np.histogram(data[i], bins)
        hists.append(hist)
    hists = np.array(hists)
    np.mean(hists, axis=0)
    delta_r = bin[1] - bin[0]
    g = []

    for k in range(bins):
        gk = hist[k] / (
            float(args.density) * (100 - 1) * np.pi * (k + 0.5) * delta_r ** 2
        )

        g.append(gk)

    return g, delta_r


read_r = read_v_and_r(save_paths["tra"])
rs = []
for frame in range(int(args.window)):
    r_ = []
    for i1, i2 in itertools.permutations(range(read_r.shape[1]), 2):
        x1 = read_r[frame, i1]
        x2 = read_r[frame, i2]

        r_.append(np.sum(toroDist3D(x1, x2, box_width) ** 2) ** 0.5)
    rs.append(r_)
rs = np.array(rs)


c, d = correlation(rs, 100)

# plotter.plot_pair_correlation()
x = np.arange(0, 100) * d
plt.plot(x, c)
plt.show()

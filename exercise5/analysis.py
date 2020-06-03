import numpy as np
import itertools

import utils
import plotter

import matplotlib.pyplot as plt


args = utils.analysisParser()
if args.eq:
    mode = "equi"
    mode_full = "equilibration"
else:
    mode = "prod"
    mode_full = "production"


save_paths = utils.createPaths(mode, args.density, args.run)

box_width = (int(100) / float(args.density)) ** 0.5


"""
######################## Task 2.2 ########################
"""
temp, e = utils.read_e_and_t(save_paths["e"])

plotter.plot_e_t(
    temp,
    tau=0.01,
    temp=True,
    title="Temperature over time, d = {}, {}".format(args.density, mode_full),
    ylabel=r"$T[K]$",
    xlabel=r"$t[ps]$",
    savepath="results/d_{}/t_time_{}.png".format(args.density, mode),
)
plotter.plot_e_t(
    e,
    tau=0.01,
    title="Energy over time, d = {}, {}".format(args.density, mode_full),
    ylabel=r"$E[u \cdot \frac{\AA^2}{ps^2}]$",
    xlabel=r"$t[ps]$",
    savepath="results/d_{}/e_time_{}.png".format(args.density, mode),
)

"""
######################## Task 2.3 ########################
"""

vel_snap, pos_snap = utils.read_v_and_r_snapshot(save_paths["snapshot"])
plotter.plot_box(
    pos_snap,
    vel_snap,
    box_width,
    xlabel=r"$x[\AA]$",
    ylabel=r"$y[\AA]$",
    title="position and velocities, d = {}".format(args.density),
    savepath="results/d_{}/snapshot.png".format(args.density),
)

"""
######################## Task 2.4 ########################
"""
if mode == "prod":
    read_v = utils.read_v_or_r(save_paths["vel"])

    v_prod = read_v[0 : int(args.window)]
    v_prod = v_prod.reshape(int(args.window) * 100, 3)
    result = np.sum(v_prod ** 2, axis=1) ** 0.5

    plotter.plot_hist(
        result,
        bins=50,
        title="velocity distribution, d = {}, #frames = {}".format(
            args.density, args.window
        ),
        ylabel="# normed",
        xlabel=r"$|\vec{v}[\AA/ps]|$",
        savepath="results/d_{}/v_distribution".format(args.density),
    )

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


if mode == "prod":
    read_r = utils.read_v_or_r(save_paths["tra"])
    rs = []
    for frame in range(int(args.window)):
        r_ = []
        for i1, i2 in itertools.permutations(range(read_r.shape[1]), 2):
            x1 = read_r[frame, i1]
            x2 = read_r[frame, i2]

            r_.append(np.sum(toroDist3D(x1, x2, box_width) ** 2) ** 0.5)
        rs.append(r_)
    rs = np.array(rs)

    g, delta_r = correlation(rs, 100)

    plotter.plot_pair_correlation(
        g,
        delta_r,
        title="Pair correlation function, d = {}".format(args.density),
        ylabel="g(r)",
        xlabel=r"$r[\AA]$",
        savepath="results/d_{}/pair_correlation".format(args.density),
    )

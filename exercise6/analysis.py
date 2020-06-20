# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import itertools


def mc_vs_theo(data, mode, value, fname):
    T = data[:, 0]
    if mode == "1D":
        if value == "U":
            mc = data[:, 2]
            theo = data[:, 4]
        if value == "C":
            mc = data[:, 3]
            theo = data[:, 5]
    if mode == "2D":
        if value == "U":
            mc = data[:, 2]
            theo = data[:, 5]
        if value == "C":
            mc = data[:, 3]
            theo = data[:, 6]
        if value == "M":
            mc = data[:, 4]
            theo = data[:, 7]
    plt.scatter(T, mc, label="MC value")
    plt.plot(T, theo, label="Analytical value")
    plt.legend()
    plt.xlabel("Temperature")
    plt.ylabel("")
    plt.title(f"{value} for {mode} model.")
    plt.grid()
    plt.savefig(f"{fname}.png")
    plt.close()


n_samples = [1000, 10000]
n_spins = [10, 100]
mode = "2D"


logs = Path("logs")

dirs = [
    logs.joinpath(mode, f"n_{n_spin}", f"ns_{n_sample}")
    for (n_spin, n_sample) in itertools.product(n_spins, n_samples)
]

for p in dirs:
    fname = p.joinpath("file")
    nspin = fname.parents[1].name
    nsamp = fname.parents[0].name
    data = np.genfromtxt(fname, skip_header=2, dtype=float)
    mc_vs_theo(data, mode, "C", f"results/C_{nspin}_{nsamp}_{mode}")
    mc_vs_theo(data, mode, "U", f"results/U_{nspin}_{nsamp}_{mode}")
    mc_vs_theo(data, mode, "M", f"results/M_{nspin}_{nsamp}_{mode}")

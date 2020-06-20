# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import numpy as np


def plot_box(r, vec, l, xlabel="", ylabel="", title="", savepath="plot_box.png"):
    r_ = r % l
    x, y = r_[:, 0], r_[:, 1]
    vx, vy = vec[:, 0], vec[:, 1]
    plt.scatter(x, y)
    for i in range(r.shape[0]):
        plt.annotate(i, (x[i], y[i]))
    plt.quiver(x, y, vx, vy)

    plt.title(title)
    axes = plt.gca()
    axes.set_xlim([0, l])
    axes.set_ylim([0, l])
    axes.set_aspect("equal", "box")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savepath)
    plt.close()


def plot_e_t(
    e_t, tau, temp=False, title="", ylabel="", xlabel="", savepath="plot_e_t.png"
):
    steps = len(e_t)
    x = np.linspace(0, steps * tau, steps)
    y = e_t
    plt.plot(x, y)
    if temp:
        plt.hlines(150, 0, steps * tau, color="red", label="Target temperatur")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_hist(
    values, bins, t=150, m=39.9, kb=0.83, title="", ylabel="", xlabel="", savepath=""
):
    n, bins, patches = plt.hist(values, bins=bins, density=True, label="data")
    x = np.linspace(0, 7, 100)
    fmb = (m / (kb * t)) * x * np.exp(-m * x ** 2 / (2 * kb * t))
    plt.plot(x, fmb, label="Maxwell-Boltzmann distribution")
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savepath)
    plt.close()


def plot_pair_correlation(g, delta_r, title="", ylabel="", xlabel="", savepath=""):
    x = np.arange(0, len(g)) * delta_r
    plt.plot(x, g)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(savepath)
    plt.close()

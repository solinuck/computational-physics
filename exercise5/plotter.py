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
        plt.hlines(150, 0, 10, color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

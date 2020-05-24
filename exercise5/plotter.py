import matplotlib.pyplot as plt
import numpy as np


def plot_box(r, v, l, lj, vel=True):
    r_ = r % l
    plt.scatter(r_[:, 0], r_[:, 1])
    for i in range(r.shape[0]):
        plt.annotate(i, (r_[i, 0], r_[i, 1]))
    if vel:
        plt.quiver(r_[:, 0], r_[:, 1], v[:, 0], v[:, 1])
    axes = plt.gca()
    x = np.arange(0, l)
    # plt.plot(x, lj.pot(x) + 10)
    axes.set_xlim([0, l])
    axes.set_ylim([0, l])
    axes.set_aspect("equal", "box")
    plt.show()


def plot_force(A, l):
    plt.quiver(A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1])  # , scale=1)
    axes = plt.gca()
    axes.set_xlim([0, l])
    axes.set_ylim([0, l])
    axes.set_aspect("equal", "box")
    plt.show()

import matplotlib.pyplot as plt

import numpy as np


def plot_box(r, vec, l, step):
    r_ = r % l
    x, y = r_[:, 0], r_[:, 1]
    vx, vy = vec[:, 0], vec[:, 1]
    plt.scatter(x, y)
    for i in range(r.shape[0]):
        plt.annotate(i, (x[i], y[i]))
    plt.quiver(x, y, vx, vy)

    plt.title("velocities  #iter = {}".format(step))
    axes = plt.gca()
    axes.set_xlim([0, l])
    axes.set_ylim([0, l])
    axes.set_aspect("equal", "box")
    plt.show()


def plot_lj(l, lj):
    x = np.arange(0, l)
    plt.plot(x, lj.pot(x) + 10)
    plt.show()

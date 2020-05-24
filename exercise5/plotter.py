import matplotlib.pyplot as plt
import numpy as np


def plot_box(r, v, l):
    r_ = r % l
    plt.scatter(r_[:, 0], r_[:, 1])
    plt.quiver([r_[:, 0], r_[:, 1]], v[:, 0], v[:, 1])
    plt.show()

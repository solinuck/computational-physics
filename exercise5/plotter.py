import matplotlib.pyplot as plt


def plot_box(r, v, l, vel=True):
    r_ = r % l
    plt.scatter(r_[:, 0], r_[:, 1])
    if vel:
        plt.quiver(r_[:, 0], r_[:, 1], v[:, 0], v[:, 1])
    plt.show()

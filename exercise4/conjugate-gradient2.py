import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def cg_method(b, x, epsilon):
    n = -multA(x) + b
    g = n
    grad = np.inf
    iter = 0
    while grad >= epsilon:
        iter += 1

        # l = g @ g
        l = (g @ g) / (n @ (multA(n)))

        x += l * n
        g_1 = g - l * multA(n)
        n_1 = g_1 + n * (g_1 @ g_1) / (g @ g)

        g = g_1
        n = n_1
        grad = np.linalg.norm(n)

        if iter == 10:
            x_10 = x.copy()
        if iter == 50:
            x_50 = x.copy()
        if iter == 100:
            x_100 = x.copy()
    return x, x_10, x_50, x_100, iter


def multA(x):
    res = np.empty_like(x)  # result vector
    width = int(np.sqrt(x.size))
    res = 4 * x  # 4 * identity
    for i in range(x.size):  # rest
        if i >= width:  # not at left edge
            res[i] -= x[i - width]
        if i < (width - 1) * width:  # not at right edge
            res[i] -= x[i + width]
        if i % width != 0:  # not at bottom edge
            res[i] -= x[i - 1]
        if (i + 1) % width != 0:  # not at top edge
            res[i] -= x[i + 1]

    return res


def create_boundary(size):
    res = np.zeros(size)
    width = int(np.sqrt(size))
    allX = np.linspace(-np.pi / 2, np.pi / 2, width)
    side = np.cos(allX)
    for i in range(size):
        if i < width:  # at left edge
            res[i] += side[i % width]
        if i >= (width - 1) * width:  # at right edge
            res[i] += side[i % width]
        if i % width == 0:  # at bottom edge
            res[i] += side[i // width]
        if (i + 1) % width == 0:  # at top edge
            res[i] += side[(i + 1) // width - 1]
    return res


def plot3D(phi, title, savename, nx=81, ny=81):
    allX = np.linspace(-np.pi / 2, np.pi / 2, nx)
    allY = np.linspace(-np.pi / 2, np.pi / 2, ny)

    xx, yy = np.meshgrid(allX, allY)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    plt.title(title)
    surf = ax.plot_surface(
        xx, yy, phi, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("{}.png".format(savename))
    plt.close()


x = np.zeros((79 ** 2))
x.fill(1)
b = create_boundary(79 ** 2)

cg_final, cg_10, cg_50, cg_100, iter = cg_method(b, x, 1e-5)

cg_final = cg_final.reshape(79, 79)
cg_10 = cg_10.reshape(79, 79)
cg_50 = cg_50.reshape(79, 79)
cg_100 = cg_100.reshape(79, 79)

print(iter)
# 131 iterations

plot3D(
    cg_final, "Task 2b: conjugate gradient final result", "img/conjugate_final", 79, 79
)
plot3D(
    cg_10, "Task 2b: conjugate gradient after 10 iterations", "img/conjugate_10", 79, 79
)
plot3D(
    cg_50, "Task 2b: conjugate gradient after 50 iterations", "img/conjugate_50", 79, 79
)
plot3D(
    cg_100,
    "Task 2b: conjugate gradient after 100 iterations",
    "img/conjugate_100",
    79,
    79,
)

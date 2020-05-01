import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def jacobi(grid, threshold):
    nx = grid.shape[0]
    ny = grid.shape[1]

    new_grid = fill_boundary(grid)
    old_grid = new_grid.copy()

    iter = 0
    while np.linalg.norm(new_grid - old_grid) > threshold or iter == 0:
        iter += 1
        old_grid = new_grid.copy()

        # loop over all inner points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                new_grid[i, j] = (
                    old_grid[i + 1, j]
                    + old_grid[i - 1, j]
                    + old_grid[i, j + 1]
                    + old_grid[i, j - 1]
                ) / 4
        if iter == 100:
            phi_100 = new_grid.copy()
    return new_grid, phi_100, iter


def fill_boundary(grid):
    nx = grid.shape[0]
    ny = grid.shape[1]
    allX = np.linspace(-np.pi / 2, np.pi / 2, nx)
    allY = np.linspace(-np.pi / 2, np.pi / 2, ny)

    # fill boundary points
    grid[[0, nx - 1], :] = np.cos(allY)
    grid = grid.T
    grid[[0, ny - 1], :] = np.cos(allX)
    grid = grid.T
    return grid


nx = 81
ny = 81
eps = 10e-5

grid = np.zeros((nx, ny))

phi_final, phi_100, iter = jacobi(grid, eps)

allX = np.linspace(-np.pi / 2, np.pi / 2, nx)
allY = np.linspace(-np.pi / 2, np.pi / 2, ny)

xx, yy = np.meshgrid(allX, allY)

fig = plt.figure()
ax = fig.gca(projection="3d")

surf = ax.plot_surface(
    xx, yy, phi_100, cmap=cm.coolwarm, linewidth=0, antialiased=False
)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig("jacobi.png")

print(iter)

from IPython import embed

embed()

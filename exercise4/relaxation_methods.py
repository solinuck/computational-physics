import numpy as np


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
    return new_grid, iter


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

relaxed, iter = jacobi(grid, eps)

print(iter)
from IPython import embed

embed()

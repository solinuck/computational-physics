import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


from matplotlib import cm


def jacobi(grid, threshold):

    new_grid = fill_boundary(grid)
    old_grid = new_grid.copy()

    iter = 0
    while np.amax(new_grid - old_grid) > threshold or iter == 0:
        iter += 1
        old_grid = new_grid.copy()

        # overwrite interior of data with interior of convolution output;
        # discard boundary. we use convolutions to quickly calculate the sum of four neighbors
        new_grid[1:-1, 1:-1] = convolve(old_grid, checkerboard((3, 3)))[1:-1, 1:-1] / 4

        if iter == 100:
            phi_100 = new_grid.copy()
    return new_grid, phi_100, iter


def gauss_seidel(grid, threshold):
    nx = grid.shape[0]
    ny = grid.shape[1]

    new_grid = fill_boundary(grid)
    old_grid = new_grid.copy()

    iter = 0
    while np.amax(new_grid - old_grid) > threshold or iter == 0:
        iter += 1
        old_grid = new_grid.copy()

        # loop over all inner points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                new_grid[i, j] = (
                    new_grid[i + 1, j]
                    + new_grid[i - 1, j]
                    + new_grid[i, j + 1]
                    + new_grid[i, j - 1]
                ) / 4
        if iter == 100:
            phi_100 = new_grid.copy()
    return new_grid, phi_100, iter


def sor(grid, threshold):
    nx = grid.shape[0]
    ny = grid.shape[1]

    new_grid = fill_boundary(grid)
    old_grid = new_grid.copy()

    # create (nx - 2)x(ny - 2) dimensional masks for the inner points.
    white = checkerboard((nx - 2, ny - 2)) == 1
    black = np.logical_not(white)

    rho_j = np.cos(np.pi / nx)
    rho_gs = rho_j ** 2
    iter = 0
    omega = 1
    threshold_reached = False
    while iter < 100:
        iter += 1

        old_grid = new_grid.copy()

        # first, only the black points are updated, the second term is again calculated with
        # convolutions, whereas only the inner black points are considered.
        new_grid[1:-1, 1:-1][black] = old_grid[1:-1, 1:-1][black] * (
            1 - omega
        ) + omega * (convolve(old_grid, checkerboard((3, 3)))[1:-1, 1:-1][black] / 4)

        # omega gets updated in between black and white
        if iter == 1:
            omega = 1 / (1 - rho_gs / 2)
        else:
            omega = omega_update(omega, rho_gs)

        # only white points are updated
        new_grid[1:-1, 1:-1][white] = new_grid[1:-1, 1:-1][white] * (
            1 - omega
        ) + omega * (convolve(new_grid, checkerboard((3, 3)))[1:-1, 1:-1][white] / 4)

        omega = omega_update(omega, rho_gs)

        if iter == 100:
            sor_100 = new_grid.copy()

        if np.amax(new_grid - old_grid) <= threshold and not threshold_reached:
            threshold_reached = True
            iter_till_threshold = iter

    return new_grid, sor_100, iter_till_threshold


def omega_update(omega, rho_GS):
    return 1 / (1 - rho_GS * omega / 4)


def fill_boundary(grid):
    nx = grid.shape[0]
    ny = grid.shape[1]
    allX = np.linspace(-np.pi / 2, np.pi / 2, nx)
    allY = np.linspace(-np.pi / 2, np.pi / 2, ny)

    # fill boundary points
    grid[[0, nx - 1], :] = np.cos(allY)  # top and bottom boundary points
    grid = grid.T  # transpose grid so that left and right are now top and bottom
    grid[[0, ny - 1], :] = np.cos(allX)  # fill the originally left and right boundary
    grid = grid.T
    return grid


def checkerboard(shape):
    """
    Accepts only tuple as input.
    Returns 2D checkerboard filled with 0 and 1
    Example: checkerboard((3,3)) -->
    [0, 1, 0]
    [1, 0, 1]
    [0, 1, 0]
    """
    return np.indices(shape).sum(axis=0) % 2


def plot3D(phi, savename, nx=81, ny=81):
    allX = np.linspace(-np.pi / 2, np.pi / 2, nx)
    allY = np.linspace(-np.pi / 2, np.pi / 2, ny)

    xx, yy = np.meshgrid(allX, allY)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    surf = ax.plot_surface(
        xx, yy, phi, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("{}.png".format(savename))
    plt.close()


nx = 81
ny = 81
eps = 1e-5

grid = np.zeros((nx, ny))

jacobi_final, jacobi_100, iter_jacobi = jacobi(grid, eps)
seidel_final, seidel_100, iter_seidel = gauss_seidel(grid, eps)
sor_final, sor_100, iter_sor = sor(grid, eps)

print("Number of iterations:")
print(f"jacobi: {iter_jacobi}")
print(f"gauss-seidel: {iter_seidel}")
print(f"successive overrelaxation: {iter_sor}")

# jacobi: 5948
# gauss-seidel: 450
# successive overrelaxation: 59

plot3D(jacobi_100, "img/jacobi_100")
plot3D(jacobi_final, "img/jacobi_final")
plot3D(seidel_100, "img/seidel_100")
plot3D(seidel_final, "img/seidel_final")
plot3D(sor_100, "img/sor_100")
plot3D(sor_final, "img/sor_final")

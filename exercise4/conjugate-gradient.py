import numpy as np


def cg_method(A, b, x, epsilon):
    n = -A @ x + b
    g = n
    grad = np.inf
    iter = 0
    while grad >= epsilon:
        iter += 1
        l = (g @ g) / (n @ (A @ n))

        x += l * n
        g_1 = g - l * A @ n
        n_1 = g_1 + n * (g_1 @ g_1) / (g @ g)

        g = g_1
        n = n_1
        grad = np.linalg.norm(n)
    return x, iter


def sd_method(A, b, x, epsilon):

    grad = np.inf
    iter = 0
    while grad >= epsilon:
        iter += 1
        n = -A @ x + b
        l = (n @ n) / (n @ (A @ n))

        x += l * n
        grad = np.linalg.norm(n)
        print(grad)
        from IPython import embed

        embed()

        if iter > 10:
            break
    return x, iter


A = np.loadtxt("CG_Matrix_10x10.dat")
b = np.zeros(10)
b.fill(1)
result, iter_sd = sd_method(A, b, b, 10e-10)
# print(cg_method(A, b, b, 1e-10))

from IPython import embed

embed()

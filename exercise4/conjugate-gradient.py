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

        x = x + l * n
        grad = np.linalg.norm(n)

        # if iter > 10:
        #     break
    return x, iter


A = np.loadtxt("CG_Matrix_10x10.dat")
b = np.zeros(10)
b.fill(1)
sd_final, iter_sd = sd_method(A, b, b, 1e-10)
cg_final, iter_cg = cg_method(A, b, b, 1e-10)

print("number of iterations:")
print("steepest descent: ", iter_sd)
print("conjugate gradient: ", iter_cg)
print("\n")
print("x[0], |x|")
print(sd_final[0], np.linalg.norm(sd_final))
print(cg_final[0], np.linalg.norm(cg_final))

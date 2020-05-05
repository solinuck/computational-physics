import numpy as np


def cg_method(b, x, epsilon):
    n = -multA(x) + b
    g = n
    grad = np.inf
    iter = 0
    while grad >= epsilon:
        iter += 1

        #l = g @ g
        l = (g @ g)/(n @ (multA(n)))

        x += l * n
        g_1 = g - l * multA(n)
        n_1 = g_1 + n * (g_1 @ g_1)/(g @ g)

        g = g_1
        n = n_1
        grad = np.linalg.norm(n)
        if iter > 10000:
            break
    return x, iter

def multA(x):
    res = np.empty_like(x) # result vector
    width = int(np.sqrt(x.size))
    res = 4 * x # 4 * identity
    for i in range(x.size): # rest
        if i >= width: # not at left edge
            res[i] -= x[i - width]
        if i < (width-1)*width: # not at right edge
            res[i] -= x[i + width]
        if i % width != 0: # not at bottom edge
            res[i] -= x[i - 1]
        if (i + 1) % width != 0: # not at top edge
            res[i] -= x[i + 1]
    return res

def create_boundary(size):
    res = np.zeros(size)#
    width = int(np.sqrt(size))
    allX = np.linspace(-np.pi / 2, np.pi / 2, width)
    side = np.cos(allX)
    for i in range(size):
        j = i % width
        if i < width: # at left edge
            res[i] += side[j]
        if i >= (width-1)*width: # at right edge
            res[i] += side[j]
        if i % width == 0: # at bottom edge
            res[i] += side[j]
        if (i + 1) % width == 0: # at top edge
            res[i] += side[j]
    return(res)


#print(multA(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])))

x = np.zeros((79**2))
x.fill(1)
b = create_boundary(79**2)


print(cg_method(b, x, 1e-5))

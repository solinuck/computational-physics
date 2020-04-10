# -*- coding: utf-8 -*-

import numpy as np


def simpson(N, a, b, func):
    points, h = np.linspace(a, b, num=N, retstep=True)
    n_slices = (N - 1) / 2
    slices = make_slices(n_slices, points)  # slices with 3 points
    integral = 0
    for idx, slice in enumerate(slices):

        # if N is even last interval must be treated differently
        if (N % 2) == 0 and idx == (len(slices) - 1):
            single_area = (
                h / 12 * (5 * func(slice[2]) + 8 * func(slice[1]) - func(slice[0]))
            )
        else:
            single_area = h / 3 * (func(slice[0]) + 4 * func(slice[1]) + func(slice[2]))
        integral += single_area
    return integral


def make_slices(n, points):
    slices = []
    for idx in range(len(points)):
        if (idx % 2) == 0 and idx != 0:
            slices += [points[idx - 2 : idx + 1]]
        # add last slice if it is an odd number
        if idx == (len(points) - 1) and len(slices) != n:
            slices += [points[idx - 2 : idx + 1]]
    return slices


function = np.sin
a = 0
b = np.pi / 2
integral = simpson(100, a, b, function)
print(integral)

import numpy as np
from numpy import random


def random_points_gen(n):
    random_points = random.rand(n, 2)
    random_points = np.append(random.normal(scale=1.0, size=n).reshape(n, 1), random.normal(scale=0.5, size=n).reshape(n, 1), axis=1)

    points = np.append(random_points, np.zeros(n).reshape(n, 1), axis=1)

    a = (random.rand() - 0.5) * 5
    b = (random.rand() - 0.5) * 5
    c = random.rand()
    points[:, 2] = a * points[:, 0] + b * points[:, 1] + c
    return points, random_points

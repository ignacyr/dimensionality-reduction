from numpy import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def points_in_circum(r, n=50):  # generate points on the circumference of a circle
    pi = math.pi
    return [[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(0, n+1)]


def points_gen(n=50, figure_type='circum'):
    points2d = {}
    # choose one type of generated points
    if figure_type == 'uniform':
        points2d = random.rand(n, 2)  # uniform distribution
    elif figure_type == 'normal':
        points2d = np.append(random.normal(scale=1.0, size=n).reshape(n, 1),  # normal distribution
                             random.normal(scale=0.5, size=n).reshape(n, 1), axis=1)
    elif figure_type == 'circum':
        points2d = np.array(points_in_circum(2.0, n-1))  # points on circle
    #
    points3d = np.append(points2d, np.zeros(n).reshape(n, 1), axis=1)
    c = random.rand()
    r = Rotation.from_euler('xyz', ((random.rand()-0.5)*90, (random.rand()-0.5)*90,
                                    (random.rand()-0.5)*90), degrees=True)  # rotation of the points on a plane
    points3d[:, 2] = c
    points3d = r.apply(points3d)
    return points3d, points2d


def plot_2d(points_2d, title=None, plot_limits=[-3, 3]):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(points_2d[:, 0], points_2d[:, 1])
    plt.xlim(plot_limits)
    plt.ylim(plot_limits)
    plt.grid()
    plt.title(title)
    ax.set_aspect('equal', adjustable='box')

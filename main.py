import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, SparsePCA
import math
from scipy.spatial.transform import Rotation

limit = 3
plot_limits = [-limit, limit]
q = 20  # quantity of random points to generate


def points_in_circum(r, n=50):  # generate points on the circumference of a circle
    pi = math.pi
    return [[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(0, n+1)]


def points_gen(n=50):
    # choose one type of generated points
    points2d = random.rand(n, 2)  # uniform distribution
    points2d = np.append(random.normal(scale=1.0, size=n).reshape(n, 1),  # normal distribution
                         random.normal(scale=0.5, size=n).reshape(n, 1), axis=1)
    points2d = np.array(points_in_circum(2.0, n-1))  # points on circle

    points3d = np.append(points2d, np.zeros(n).reshape(n, 1), axis=1)
    c = random.rand()
    r = Rotation.from_euler('xyz', ((random.rand()-0.5)*90, (random.rand()-0.5)*90,
                                    (random.rand()-0.5)*90), degrees=True)  # rotation of the points on a plane
    points3d[:, 2] = c
    points3d = r.apply(points3d)
    return points3d, points2d


def plot_2d(points_2d, title):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(points_2d[:, 0], points_2d[:, 1])
    plt.xlim(plot_limits)
    plt.ylim(plot_limits)
    plt.grid()
    plt.title(title)
    ax.set_aspect('equal', adjustable='box')


def main():
    # generating random points
    random_points_3d, random_points_2d = points_gen(q)

    # 3D plot
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(projection='3d')
    ax3d.scatter(random_points_3d[:, 0], random_points_3d[:, 1], random_points_3d[:, 2])
    plt.title('3D')
    ax3d.set_xlim3d(plot_limits)
    ax3d.set_ylim3d(plot_limits)
    ax3d.set_zlim3d(plot_limits)

    # Original points on 2d plane plot
    plot_2d(random_points_2d, 'Original')

    # PCA points on 2d plane plot
    pca = PCA(n_components=2)
    points_2d_pca = pca.fit_transform(random_points_3d)
    plot_2d(points_2d_pca, 'PCA')

    # Truncated SVD (LSA) points on 2d plane plot
    svd = TruncatedSVD(n_components=2)
    points_2d_svd = svd.fit_transform(random_points_3d)
    plot_2d(points_2d_svd, 'Truncated SVD')

    # Kernel PCA
    k_pca = KernelPCA(n_components=2)
    points_2d_k_pca = k_pca.fit_transform(random_points_3d)
    plot_2d(points_2d_k_pca, 'Kernel PCA')

    # Sparse PCA
    s_pca = SparsePCA(n_components=2)
    points_2d_s_pca = s_pca.fit_transform(random_points_3d)
    plot_2d(points_2d_s_pca, 'Sparse PCA')

    plt.show()


if __name__ == '__main__':
    main()

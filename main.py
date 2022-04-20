import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, SparsePCA
import points_gen_3d as pg3d


def main():
    limit = 3
    plot_limits = [-limit, limit]
    q = 20  # quantity of random points to generate

    # generating random points
    random_points_3d, random_points_2d = pg3d.points_gen(q)

    # 3D plot
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(projection='3d')
    ax3d.scatter(random_points_3d[:, 0], random_points_3d[:, 1], random_points_3d[:, 2])
    plt.title('3D')
    ax3d.set_xlim3d(plot_limits)
    ax3d.set_ylim3d(plot_limits)
    ax3d.set_zlim3d(plot_limits)

    # Original points on 2d plane plot
    pg3d.plot_2d(random_points_2d, 'Original', plot_limits)

    # PCA points on 2d plane plot
    pca = PCA(n_components=2)
    points_2d_pca = pca.fit_transform(random_points_3d)
    pg3d.plot_2d(points_2d_pca, 'PCA', plot_limits)

    # Truncated SVD (LSA) points on 2d plane plot
    svd = TruncatedSVD(n_components=2)
    points_2d_svd = svd.fit_transform(random_points_3d)
    pg3d.plot_2d(points_2d_svd, 'Truncated SVD', plot_limits)

    # Kernel PCA
    k_pca = KernelPCA(n_components=2)
    points_2d_k_pca = k_pca.fit_transform(random_points_3d)
    pg3d.plot_2d(points_2d_k_pca, 'Kernel PCA', plot_limits)

    # Sparse PCA
    s_pca = SparsePCA(n_components=2)
    points_2d_s_pca = s_pca.fit_transform(random_points_3d)
    pg3d.plot_2d(points_2d_s_pca, 'Sparse PCA')

    plt.show()


if __name__ == '__main__':
    main()

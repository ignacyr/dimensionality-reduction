import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from random_points import *

limit = 3
plot_limits = [-limit, limit]
n = 10
random_points_3d, random_points_2d = random_points_gen(n)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(random_points_3d[:, 0], random_points_3d[:, 1], random_points_3d[:, 2])
plt.title('3D')
ax.set_xlim3d(plot_limits)
ax.set_ylim3d(plot_limits)
ax.set_zlim3d(plot_limits)

pca = PCA(n_components=2)
points_2d_pca = pca.fit_transform(random_points_3d)

svd = TruncatedSVD(n_components=2)
points_2d_svd = svd.fit_transform(random_points_3d)


# Original points on 2d plane plot
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(random_points_2d[:, 0], random_points_2d[:, 1])
plt.xlim(plot_limits)
plt.ylim(plot_limits)
plt.grid()
plt.title('Original')

# PCA points on 2d plane plot
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(points_2d_pca[:, 0], points_2d_pca[:, 1])
plt.xlim(plot_limits)
plt.ylim(plot_limits)
plt.grid()
plt.title('PCA')

# SVD points on 2d plane plot
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(points_2d_svd[:, 0], points_2d_svd[:, 1])
plt.xlim(plot_limits)
plt.ylim(plot_limits)
plt.grid()
plt.title('SVD')

plt.show()



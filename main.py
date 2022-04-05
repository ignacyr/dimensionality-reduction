import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from random_points import *

plot_limits = [-5, 5]
n = 10
random_points_3d, random_points_2d = random_points_gen(n)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(random_points_3d[:, 0], random_points_3d[:, 1], random_points_3d[:, 2])


pca = PCA(n_components=2)
points_2d = pca.fit_transform(random_points_3d)

print(random_points_2d.shape)
print(random_points_3d.shape)
print(points_2d.shape)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(points_2d[:, 0], points_2d[:, 1])
plt.xlim(plot_limits)
plt.ylim(plot_limits)
plt.grid()
plt.title('PCA')

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(random_points_2d[:, 0], random_points_2d[:, 1])
plt.xlim(plot_limits)
plt.ylim(plot_limits)
plt.grid()
plt.title('Original')



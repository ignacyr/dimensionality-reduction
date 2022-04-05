import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from sklearn.decomposition import PCA

n = 10
random_points = random.rand(n, 2)
random_points = np.append(random.normal(scale=1.0, size=n).reshape(n, 1), random.normal(scale=0.5, size=n).reshape(n, 1), axis=1)

points = np.append(random_points, np.zeros(n).reshape(n, 1), axis=1)

A = (random.rand() - 0.5) * 5
B = (random.rand() - 0.5) * 5
C = random.rand()
points[:, 2] = A * points[:, 0] + B * points[:, 1] + C

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])


pca = PCA(n_components=2)
points2 = pca.fit_transform(points)

print(random_points.shape)
print(points.shape)
print(points2.shape)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(points2[:, 0], points2[:, 1])
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.grid()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(random_points[:, 0], random_points[:, 1])
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.grid()




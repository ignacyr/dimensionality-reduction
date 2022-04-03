import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from sklearn.decomposition import PCA

n = 300
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

print(points2.shape)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(points2[:, 0], points2[:, 1])





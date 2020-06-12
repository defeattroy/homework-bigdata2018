from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
X, y = make_blobs(n_samples=150, n_features=2, centers=3,cluster_std=0.5, shuffle=True, random_state=0)
#转换成numpy array
X = np.array(X)
#类簇数量
n_clusters = 3
#聚类
cls = KMeans(n_clusters).fit(X)
#X中每项所属分类的一个列表
cls.labels_
for i in range(n_clusters):
	members = cls.labels_ == i
	plt.scatter(X[members, 0], X[members, 1], s=60,  c='b', alpha=0.5)
plt.title(' ')
plt.show()


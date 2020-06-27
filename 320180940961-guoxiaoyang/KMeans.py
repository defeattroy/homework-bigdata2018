from sklearn.datasets import make_blobs  # 导入产生模拟数据的方法
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# 1. 产生模拟数据
k = 3
X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)

# 2. 模型构建
km = KMeans(n_clusters=k, init='k-means++', max_iter=30)
km.fit(X)

# 获取簇心
centroids = km.cluster_centers_
# 获取归集后的样本所属簇对应值
y_kmean = km.predict(X)

# 呈现未归集前的数据
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.yticks(())
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_kmean, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, alpha=0.5)
plt.show()
#参考：https://blog.csdn.net/u013850277/java/article/details/88411966

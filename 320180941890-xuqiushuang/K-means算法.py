import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)
print(f'Shape of dataset: {X.shape}')

fig = plt.figure(figsize=(12,10))
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Dataset with 3 clusters")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
class KMeans():
    def __init__(self, n_clusters=3):
        self.k = n_clusters

    def fit(self, data):#适用于给定数据集
        n_samples, _ = data.shape
        # 初始化聚类中心
        self.centers = np.array(random.sample(list(data), self.k))
        self.initial_centers = np.copy(self.centers)

        #我们将跟踪数据点的分配是否改变。如果它停止变化，我们完成拟合模型
        old_assigns = None
        n_iters = 0

        while True:
            new_assigns = [self.classify(datapoint) for datapoint in data]

            if new_assigns == old_assigns:
                print(f"Training finished after {n_iters} iterations!")
                return

            old_assigns = new_assigns
            n_iters += 1

            # 重新计算中心
            for id_ in range(self.k):
                points_idx = np.where(np.array(new_assigns) == id_)
                datapoints = data[points_idx]
                self.centers[id_] = datapoints.mean(axis=0)

    def l2_distance(self, datapoint):
        dists = np.sqrt(np.sum((self.centers - datapoint)**2, axis=1))
        return dists

    def classify(self, datapoint):#给定一个数据点，计算离数据点最近的簇。 返回该群集的群集ID。
        dists = self.l2_distance(datapoint)
        return np.argmin(dists)

    def plot_clusters(self, data):
        plt.figure(figsize=(12,10))
        plt.title("Initial centers in black, final centers in red")
        plt.scatter(data[:, 0], data[:, 1], marker='.', c=y)
        plt.scatter(self.centers[:, 0], self.centers[:,1], c='r')
        plt.scatter(self.initial_centers[:, 0], self.initial_centers[:,1], c='k')
        plt.show()
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans.plot_clusters(X)

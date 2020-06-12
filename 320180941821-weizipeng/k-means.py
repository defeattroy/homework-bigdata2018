import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans#引入kmeans


## 产生模拟数据
N = 150
centers = 3
data,y = ds.make_blobs(N, n_features=2, centers=centers, random_state=28)
data2,y2 = ds.make_blobs(N, n_features=2, centers=centers,  random_state=28)
data3 = np.vstack((data[y == 0][:200], data[y == 1][:100], data[y == 2][:10], data[y == 3][:50]))
y3 = np.array([0] * 200 + [1] * 100 + [2] * 10 + [3] * 50)



#一、数据前期处理跟前面模型是一样
#二、模型的构建
km = KMeans(n_clusters=centers, init='random',random_state=28)
#n_clusters就是K值，也是聚类值
#init初始化方法
km.fit(data, y)

y_hat = km.predict(data)
print ("所有样本距离聚簇中心点的总距离和:", km.inertia_)
print ("距离聚簇中心点的平均距离:", (km.inertia_ / N))
cluster_centers = km.cluster_centers_
print ("聚簇中心点：", cluster_centers)

y_hat2 = km.fit_predict(data2)
y_hat3 = km.fit_predict(data3)

def expandBorder(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


cm = mpl.colors.ListedColormap(list('rgbmyc'))
plt.figure(figsize=(15, 9), facecolor='w')
plt.subplot(241)
plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')

x1_min, x2_min = np.min(data, axis=0)
x1_max, x2_max = np.max(data, axis=0)
x1_min, x1_max = expandBorder(x1_min, x1_max)
x2_min, x2_max = expandBorder(x2_min, x2_max)
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'原始数据')
plt.grid(True)

plt.subplot(242)
plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'K-Means算法聚类结果')
plt.grid(True)

plt.show()
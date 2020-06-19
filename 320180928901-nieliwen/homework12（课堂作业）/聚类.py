#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans#引入kmeans

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# In[33]:


N = 1500 # 1500个样本
centers = 4 # 4个聚簇中心点
data,y = ds.make_blobs(N, n_features=2, centers=centers, random_state=28)

#data3 = np.vstack((data[y == 0][:200], data[y == 1][:100],
     #data[y == 2][:10], data[y == 3][:50]))
#y3 = np.array([0] * 200 + [1] * 100 + [2] * 10 + [3] * 50)


# In[34]:


km = KMeans(n_clusters=centers, init='random',random_state=28)
km.fit(data, y)


# In[35]:


y_hat = km.predict(data)
print ("所有样本距离聚簇中心点的总距离和:", km.inertia_)
print ("距离聚簇中心点的平均距离:", (km.inertia_ / N))
cluster_centers = km.cluster_centers_
print ("聚簇中心点：", cluster_centers)


# In[36]:


y_hat2 = km.fit_predict(data2)
y_hat3 = km.fit_predict(data3)


# In[37]:


def expandBorder(a, b):
    d = (b - a) * 0.1
    return a-d, b+d
cm = mpl.colors.ListedColormap(list('rgbmyc'))
plt.figure(figsize=(15, 9), facecolor='w')


# In[38]:


#原始数据
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


# In[39]:


#K-Means算法聚类结果
plt.subplot(242)
plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'K-Means算法聚类结果')
plt.grid(True)

m = np.array(((1, 1), (0.5, 5)))
data_r = data.dot(m)
y_r_hat = km.fit_predict(data_r)


# In[ ]:





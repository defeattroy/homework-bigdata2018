"""
数据科学与大数据技术
胡叶龙
320180941011
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


X,y = make_blobs(n_samples=150,
				 n_features=2,
				 centers=3,
				 cluster_std=0.5,
				 shuffle=True,
				 random_state=0)

def distEclud(vecA, vecB):
	#定义一个欧式距离的函数  
	return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

from sklearn.cluster import KMeans
sse = []
for k in range(2,26):
	km = KMeans(n_clusters = k, random_state = 100)
	km.fit(X)

	sse.append(km.inertia_)

plt.plot(range(2,26), sse, marker = 'o')
plt.xlabel("k value")
plt.ylabel("SSE")
plt.show()
#可得k取3最好

#随机设置k个中心点
def randCent(dataSet, k):
	#第一个中心点初始化
	n = np.shape(dataSet)[1]
	centroids = np.mat(np.zeros([k, n])) 
	#创建k行n列的全为0的矩阵
	for j in range(n):
		minj = np.min(dataSet[:,j]) #获得第j列的最小值
		rangej = float(np.max(dataSet[:,j]) - minj)  #得到最大值与最小值之间的范围
		#获得输出为K行1列的数据，并且使其在数据集范围内
		centroids[:,j] = np.mat(minj + rangej * np.random.rand(k, 1))   
	return centroids

def KMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
	#参数： dataSet：样本点，k：簇的个数
	#disMeans：距离，使用欧式距离，createCent：初始中心点的选取
	m = np.shape(dataSet)[0]    #得到行数，即为样本数
	clusterAssement = np.mat(np.zeros([m,2]))   #创建m行2列的矩阵
	centroids = createCent(dataSet, k)      #初始化k个中心点
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = np.inf   #初始设置值为无穷大
			minIndex = -1
			for j in range(k):
				#j循环，先计算k个中心点到1个样本的距离，在进行i循环，计算得到k个中心点到全部样本点的距离
				distJ = distMeans(centroids[j,:], dataSet[i,:])
				if distJ <  minDist:
					minDist = distJ #更新最小的距离
					minIndex = j 
			if clusterAssement[i,0] != minIndex:    #如果中心点不变化的时候，则终止循环
				clusterChanged = True 
			clusterAssement[i,:] = minIndex, minDist**2 #将index，k值中心点和最小距离存入到数组中
		#print(centroids)
		
		#更换中心点的位置
		for cent in range(k):
			ptsInClust = dataSet[np.nonzero(clusterAssement[:,0].A == cent)[0]] #分别找到属于k类的数据
			centroids[cent,:] = np.mean(ptsInClust, axis = 0)   #得到更新后的中心点
		
	return centroids, clusterAssement 

center, cluster = KMeans(X, 3)
print("中心点:\n",center)
print("聚类结果:\n",cluster)
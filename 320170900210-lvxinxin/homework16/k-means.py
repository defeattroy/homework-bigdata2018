from sklearn.datasets import make_blobs
from numpy import *
import matplotlib.pyplot as plt

X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)


# 欧氏距离计算
def distEclud(x,y):
    return sqrt(sum(power(x-y,2)))  # 计算欧氏距离

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = mat(zeros((k,n)))
    for i in range(k):
        index = int(random.uniform(0,m)) #
        centroids[i,:] = dataSet[index,:]
    return centroids

# k均值聚类
def KMeans(dataSet,k):
    m = shape(dataSet)[0]  #行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = mat(zeros((m,2)))
    clusterChange = True
    #初始化centroids
    centroids = randCent(dataSet,k)
    while clusterChange:
        clusterChange = False
        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            #找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            #更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        #更新质心
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
            centroids[j,:] = mean(pointsInCluster,axis=0)   # 对矩阵的行求均值
    print("Congratulations,cluster complete!")
    return centroids,clusterAssment

datMat = mat(X)
myCentroids,clustAssing = KMeans(datMat,3)
# myCentroids中心点坐标,clustAssing是分类结果
print('中心点坐标是：')
print(myCentroids)

# 绘制数据分布图
plt.scatter(X[:,0],X[:,1],c = y)
plt.scatter(myCentroids[0,0],myCentroids[0,1], c = "black", marker='^')
plt.scatter(myCentroids[1,0],myCentroids[1,1], c = "black", marker='^')
plt.scatter(myCentroids[2,0],myCentroids[2,1], c = "black", marker='^')plt.show()

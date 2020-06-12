from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)
import numpy as np
from matplotlib import pyplot as plt
plt.scatter(X[:,0], X[:,1])
plt.show()

class My_KMeans:
    def __init__(self,k,dataSet):
        self.k = k
        self.dataSet = dataSet
        #种类数目和数据集
        
    #计算某个点和簇中心的距离    
    def distEclud(self,x,centre):
        return np.sqrt(np.sum((x-centre)**2))  # 计算欧氏距离
    
    # 为给定数据集构建一个包含K个随机质心的集合
    def randCent(self):
        m,n = self.dataSet.shape
        centroids = np.zeros((self.k,n))
        for i in range(self.k):
            index = int(np.random.uniform(0,m)) #
            centroids[i,:] =  self.dataSet[index,:]
        return centroids
    
    # k均值聚类
    def KMeans(self):
 
        m = np.shape(self.dataSet)[0]  #行的数目
        # 第一列存样本属于哪一簇
        # 第二列存样本的到簇的中心点的误差
        clusterAssment = np.mat(np.zeros((m,2)))
        clusterChange = True
 
        # 第1步 初始化centroids
        centroids = self.randCent()
        while clusterChange:
            clusterChange = False
 
            # 遍历所有的样本（行数）
            for i in range(m):
                minDist = 100000.0
                minIndex = -1
 
                # 遍历所有的质心
                #第2步 找出最近的质心
                for j in range(self.k):
                    # 计算该样本到质心的欧式距离
                    distance = self.distEclud(centroids[j,:],self.dataSet[i,:])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                # 第 3 步：更新每一行样本所属的簇
                if clusterAssment[i,0] != minIndex:
                    clusterChange = True
                    clusterAssment[i,:] = minIndex,minDist**2
            #第 4 步：更新质心
            for j in range(self.k):
                pointsInCluster =  self.dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
                centroids[j,:] = np.mean(pointsInCluster,axis=0)   # 对矩阵的行求均值
 
        print("Congratulations,cluster complete!")
        return centroids,clusterAssment
    
    def showCluster(self):
        centroids,clusterAssment = self.KMeans()
        m,n = dataSet.shape
        if n != 2:
            print("数据不是二维的")
            return 1
 
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        if self.k > len(mark):
            print("k值太大了")
            return 1
 
        # 绘制所有的样本
        for i in range(m):
            markIndex = int(clusterAssment[i,0])
            plt.plot(dataSet[i,0],self.dataSet[i,1],mark[markIndex])
 
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 绘制质心
        for i in range(self.k):
            plt.plot(centroids[i,0],centroids[i,1],mark[i])
 
        plt.show()
    

Example = My_KMeans(3,X)
Example.showCluster()

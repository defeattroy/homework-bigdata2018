#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import make_blobs
from matplotlib import pyplot
X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)

color = ['red', 'pink','orange']
fig, axi1=plt.subplots(1)
for i in range(3):
    axi1.scatter(X[y==i, 0], X[y==i,1],
               marker='o',
               s=8,
               c=color[i]
               )
plt.show()

#摘自https://blog.csdn.net/wuxiaosi808/article/details/90205065?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.nonecase
def loadDataSet(fileName):      #分析制表符分隔的浮点数的通用函数
    dataMat = []                #假设最后一列是目标值
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #将所有元素映射到float（）
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#创建 centroid mat
    for j in range(n):#在每个维度的范围内创建随机簇中心
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #样本数
    clusterAssment = mat(zeros((m,2))) #m*2的矩阵                   
    centroids = createCent(dataSet, k) #初始化k个中心
    clusterChanged = True             
    while clusterChanged:      #当聚类不再变化
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k): #找到最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            # 第1列为所属质心，第2列为距离
            clusterAssment[i,:] = minIndex,minDist**2
        print (centroids)

        # 更改质心位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    return centroids, clusterAssment


# In[ ]:





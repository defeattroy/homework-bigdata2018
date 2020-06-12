#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

'''标志位统计递归运行次数'''
flag = 0

'''欧式距离'''
def ecludDist(x, y):
    return np.sqrt(sum(np.square(np.array(x) - np.array(y))))

'''计算簇的均值点'''
def clusterMean(dataset):
    return sum(np.array(dataset)) / len(dataset)

'''生成随机均值点'''
def randCenter(dataset, k):
    temp = []
    while len(temp) < k:
        index = np.random.randint(0, len(dataset)-1)
        if  index not in temp:
            temp.append(index)  
    #temp仅包含随机均值点的索引,为三个
    return np.array([dataset[i] for i in temp])

def kMeans(dataset, dist, center, k):
    '''
    dataset = X
    dist = ecludDist()  欧式距离
    center = randCenter  生成随机均值点
    k = 3
    '''
    global flag
    #all_kinds用于存放中间计算结果
    all_kinds = []
    for _ in range(k):
        temp = []
        all_kinds.append(temp)
    #计算每个点到各随机均值点的距离
    for i in dataset:
        temp = []
        for j in center:  # center 已包含三个随机均值点的坐标
            temp.append(dist(i, j))
        all_kinds[temp.index(min(temp))].append(i)   # 将该节点归类到距离最近的随机均值点所在的列表里
        
    flag += 1
    #更新均值点：分别在分好的三类数据集当中操作第一次程序
    center_ = np.array([clusterMean(i) for i in all_kinds])
    if (center_ == center).all():  # 判断是否迭代完成
        for i in range(k):
            plt.scatter([j[0] for j in all_kinds[i]], [j[1] for j in all_kinds[i]], marker='*')
        plt.grid()
        plt.show()
    else:
        #递归调用kMeans函数
        center = center_
        kMeans(dataset, dist, center, k)

def main(k):
    '''生成随机点''' 
    X,y = make_blobs(n_samples=60,
                 n_features=2, #维数
                 centers=3, #//产生数据的中心点，默认值3
                 cluster_std=1.0,# cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0
                 shuffle=True,# shuffle ：洗乱，默认值是True
                 random_state=0)  #随机生成器的种子
    X.tolist()
    plt.plot(X[:,0],X[:,1],'b.')
    plt.show()
    initial_center = randCenter(dataset=X, k=k)
    kMeans(dataset=X, dist=ecludDist, center=initial_center, k=k)

if __name__ == '__main__':
    main(3) 


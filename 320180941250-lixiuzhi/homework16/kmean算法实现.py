#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np

class KMeans:
    def __init__(self,k_clusters,tol=1e-4,max_iter=300):
        self.k_clusters = k_clusters  #分为k簇
        self.tol = tol  #用于判断算法是否收敛
        self.max_iter = max_iter   #最大迭代次数
        
    def _init_centers_random(self,x,k_clusters):
        #随机产生k个簇的中心点
        p, n = X.shape
        xmax = np.max(X, axis=0)
        xmin = np.min(X, axis=0)#获取特征的取值范围
        
        return xmin + (xmax - xmin) * np.random.rand(k_culsters, n) #在特征范围内，使用均匀分布产生k个簇的中心点
        
        
    def kmeans(self,X):
        m, n = X.shape
        labels = np.zeros(m, dtype=np.int)#存储对m个实例划分簇的标记
        distances = np.empty((m,self.k_clusters)) #存储矩阵，存储m个实例到k个中心点的距离
        centers_old = np.empty(self.k_clusters,n)#存储之前的簇中心点
        
        
        centers = self._init_centers_random(X,self.k_clusters)#初始化簇的中心点
        
        for p in range(self.max_iter):
            for i in range(self.k_clusters):
                np.sum((X - centers[i]) ** 2,axis=1,out=distances[:,i])# 计算m个实例到各中心点的距离
                
            np.argmin(distances,axis=1,out=labels) #将m个实例划分到最近的中心点区域
            
            np.copyto(centers_old,centers)#保存之前的簇中心点
            for i in range(self.k_clusters):
                cluster = X[labels == 1]#得到某簇的所有数据
                if cluster.size == 0:#若某个初始中心点离所有数据都很远导致没有实例划分进
                    return None  #则没有被划分为k簇，划分失败返回none值
                np.mean(cluster, axis=0,out=centers[i])  #使用重新划分的簇计算簇中心点
            
            delta_centers = np.sqrt(np.sum((centers - centers_old) ** 2,axis=1))  #计算新中心点和旧中心点的距离
            
            if np.all(delta_centers < self.tol):
                break  #距离低于tol则判为算法收敛，结束迭代
                
        return labels, centers
    
    def predict(self,X):
        result = None
        while not result:
            result = self._kmeans(X)  #调用self._kmeans直到成功划分
            
        labels, self.centers_ = result  #将划分标记返回，并将簇中心点保存到类属性
        
        
        return labels
        
        
        







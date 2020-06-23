from sklearn.datasets import make_blobs
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt

X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)

def distEclud(vecA, vecB):#计算欧式距离
    vecA= np.array(vecA)
    vecB= np.array(vecB)
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def means(a):#元组列表的平均值
    m=[]
    n=[]
    for i in a:
        m.append(i[0])
        n.append(i[1])
    m=np.mean(m)
    n=np.mean(n)
    return (m,n)
    
def center(cluster_dict):#构建聚类中心
    a= [n for n,k in cluster_dict.items() if k==0]
    b= [n for n,k in cluster_dict.items() if k==1]
    c= [n for n,k in cluster_dict.items() if k==2]
    return {means(a):0,means(b):1,means(c):2}
           
def classify(center_dict,a):#分类
    tem={}
    for i in center_dict.keys():
        tem[center_dict[i]]=distEclud(a,i)
    return min(tem,key=tem.get)#返回最小值对应的区域

def turn(X):#转化数据集类型
    my_list=[]
    X=list(X)
    for a in X:
        a=tuple(a)
        my_list.append(a)
    return my_list
        
def Kmeans(X):#数据集
    X=turn(X)#转换数据类型
    cluster_dict={}#分类好的群组的字典
    center_dict={}#中心的群组的字典
    n_cluster=3#分类数
    #寻找最初的中心点
    for i in range(0,n_cluster):
        a=random.randint(0,len(X))
        if X[a] not in center_dict:
            center_dict[X[a]]=i
        else:
            i=i-1
    #迭代聚类
    for x in range(0,20):#最高迭代次数为10次
        new_center_dict={}
        for tem in X:#对所有元素分类
            cluster_dict[tem]=classify(center_dict,tem)
        #重新构建聚类中心
        new_center_dict=center(cluster_dict)
        if new_center_dict==center_dict:#如果两次聚类中心相等，说明已经收敛
            break
        else:
            center_dict=new_center_dict
    return cluster_dict
                
a=Kmeans(X)
i= [list(n) for n,k in a.items() if k==0]
i1=[x[0] for x in i]
i2=[x[1] for x in i]
j= [list(n) for n,k in a.items() if k==1]
j1=[x[0] for x in j]
j2=[x[1] for x in j]
k= [list(n) for n,k in a.items() if k==2]
k1=[x[0] for x in k]
k2=[x[1] for x in k]

plt.scatter(i1,i2,c = "red", marker='o', label='label0')
plt.scatter(j1,j2, c = "green", marker='*', label='label1')
plt.scatter(k1,k2, c = "blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
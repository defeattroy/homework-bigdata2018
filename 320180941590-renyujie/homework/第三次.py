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

def distEclud(vecA, vecB):#����ŷʽ����
    vecA= np.array(vecA)
    vecB= np.array(vecB)
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def means(a):#Ԫ���б��ƽ��ֵ
    m=[]
    n=[]
    for i in a:
        m.append(i[0])
        n.append(i[1])
    m=np.mean(m)
    n=np.mean(n)
    return (m,n)
    
def center(cluster_dict):#������������
    a= [n for n,k in cluster_dict.items() if k==0]
    b= [n for n,k in cluster_dict.items() if k==1]
    c= [n for n,k in cluster_dict.items() if k==2]
    return {means(a):0,means(b):1,means(c):2}
           
def classify(center_dict,a):#����
    tem={}
    for i in center_dict.keys():
        tem[center_dict[i]]=distEclud(a,i)
    return min(tem,key=tem.get)#������Сֵ��Ӧ������

def turn(X):#ת�����ݼ�����
    my_list=[]
    X=list(X)
    for a in X:
        a=tuple(a)
        my_list.append(a)
    return my_list
        
def Kmeans(X):#���ݼ�
    X=turn(X)#ת����������
    cluster_dict={}#����õ�Ⱥ����ֵ�
    center_dict={}#���ĵ�Ⱥ����ֵ�
    n_cluster=3#������
    #Ѱ����������ĵ�
    for i in range(0,n_cluster):
        a=random.randint(0,len(X))
        if X[a] not in center_dict:
            center_dict[X[a]]=i
        else:
            i=i-1
    #��������
    for x in range(0,20):#��ߵ�������Ϊ10��
        new_center_dict={}
        for tem in X:#������Ԫ�ط���
            cluster_dict[tem]=classify(center_dict,tem)
        #���¹�����������
        new_center_dict=center(cluster_dict)
        if new_center_dict==center_dict:#������ξ���������ȣ�˵���Ѿ�����
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

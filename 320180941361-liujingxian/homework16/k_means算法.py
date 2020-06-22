'''
from sklearn.datasets import make_blobs
from matplotlib import pyplot

X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)

#在2D图中绘制样本，每个样本颜色不同
pyplot.scatter(X[:,0],X[:,1],c=y);
pyplot.show()
'''


from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

'''
# 导入数据集
data = pd.read_csv('xclara.csv')
# print(data.shape)
# data.head()
'''

X,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)
data=X,y

f1 = X[:,0]
f2 = X[:,1]

X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=6)



# 按行的方式计算两个坐标点之间的距离
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# 设定分区数
k = 3
# 随机获得中心点的X轴坐标
C_x = np.random.randint(0, np.max(X)-1, size=k)
# 随机获得中心点的Y轴坐标
C_y = np.random.randint(0, np.max(X)-1, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)


# 将初始化中心点画到输入的样例数据上
plt.scatter(f1, f2, c='black', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='red')


# 用于保存中心点更新前的坐标
C_old = np.zeros(C.shape)
print(C)
# 用于保存数据所属中心点
clusters = np.zeros(len(X))
# 迭代标识位，通过计算新旧中心点的距离
iteration_flag = dist(C, C_old, 1)

tmp = 1
# 若中心点不再变化或循环次数不超过20次(此限制可取消)，则退出循环
while iteration_flag.any() != 0:
    # 循环计算出每个点对应的最近中心点
    for i in range(len(X)):
        # 计算出每个点与中心点的距离
        distances = dist(X[i], C, 1)
        # print(distances)
        # 记录0 - k-1个点中距离近的点
        cluster = np.argmin(distances) 
        # 记录每个样例点与哪个中心点距离最近
        clusters[i] = cluster
        
    # 采用深拷贝将当前的中心点保存下来
    # print("the distinct of clusters: ", set(clusters))
    C_old = deepcopy(C)
    # 从属于中心点放到一个数组中，然后按照列的方向取平均值
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        # print(points)
        # print(np.mean(points, axis=0))
        C[i] = np.mean(points, axis=0)
        # print(C[i])
    # print(C)
    
    # 计算新旧节点的距离
    print ('循环第%d次' % tmp)
    tmp = tmp + 1
    iteration_flag = dist(C, C_old, 1)
    print("新中心点与旧点的距离：", iteration_flag)

# 最终结果图示
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
# 不同的子集使用不同的颜色
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')
plt.show()
#展示生成的数据集原本的分类
plt.scatter(X[:,0],X[:,1],c=y);
plt.show()

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

iris=datasets.load_iris()
x=iris.data
y=iris.target

def kmeans(data, n, m, k, plt):
    """数据集中随机索引"""
    rarray = np.random.random(size=k)
    rarray = np.floor(rarray*n)
    rarray = rarray.astype(int)
    """初始center"""
    center = data[rarray]
    cls = np.zeros([n], np.int)
    """new center"""
    run = True
    time = 0
    while run:
        time = time + 1
        for i in range(n):
            tmp = data[i] - center
            tmp = np.square(tmp)
            tmp = np.sum(tmp, axis=1)
            cls[i] = np.argmin(tmp)
        run = False
        for i in range(k):
            club = data[cls==i]
            newcenter = np.mean(club, axis=0)
            ss = np.abs(center[i]-newcenter)
            if np.sum(ss, axis=0) > 1e-4:
                center[i] = newcenter
                run = True
    print('程序结束，迭代次数：', time)
    for i in range(k):
        club = data[cls == i]
        showtable(club, plt)
    showtable(center, plt)


def showtable(data, plt):
    x = data.T[0]
    y = data.T[1]
    plt.scatter(x, y)

kmeans(x, 150, 4, 3, plt)
plt.show()

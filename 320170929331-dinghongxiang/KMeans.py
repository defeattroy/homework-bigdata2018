# -*- coding: utf-8 -*-

"""

@Time: 2020/6/12 12:20

@Auth: Erris

@Version:  Python 3.8.0


"""

import numpy as np
import random
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

class MyKMeans():
    '''class MyKMeans
    usage:
    clf = MyKMeans([K=...][, max_iter=...])
    clf.train(data)
    '''
    
    def __init__(self, K=3, max_iter=500):
    # initialize
    # K: number of clusters
    # max_iter: maximum number to stop the iteration
        self.K = K
        self.centers = []
        self.max_iter = max_iter
        
    def distance(self, dotA, dotB):
    # calculate the distance between dot A and B
        dis = 0
        for i in range(len(dotA)):
            dis += (dotA[i]-dotB[i])**2
        return np.sqrt(dis)
    
    def classify(self, centers, samp):
    # classify the sample by distance
        return np.argmin([self.distance(x, samp) for x in centers])
    
    def newCenter(self, data, clusters):
    # calculate the new center
        centers_new = []
        for Ki in range(self.K):
            counter = 0
            center_tmp = np.zeros((1, len(data[0])))[0]
            for i in range(len(clusters)):
                if clusters[i] == Ki:
                    counter+=1
                    center_tmp += np.array(data[i])
            center_tmp/=counter
            centers_new.append(list(center_tmp))
        return centers_new
                
    def train(self, data):
    #train the data and return clusters and centers
        clusters = [0 for i in data]
        self.centers = data[random.sample(range(len(data)), self.K)]
        '''
        while True:
            self.centers = data[random.sample(range(len(data)), self.K)]
            for i in range(len(self.centers)):
                for j in range(i, len(self.centers)):
                    if self.distance(self.centers[i], self.centers[j])<4:
                        continue
            break
        '''
        for it in range(self.max_iter):
            for i in range(len(data)):
                clusters[i] = self.classify(self.centers, data[i])
            if (self.centers == self.newCenter(data, clusters)).any():
                break
            else: self.center = self.newCenter(data, clusters)
        return clusters, self.centers
        
X,y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

kMeans = MyKMeans()
clusters, centers = kMeans.train(X)

colors = ['c', 'b', 'r']

for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], color = colors[clusters[i]])
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()
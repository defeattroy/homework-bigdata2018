import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
data = np.random.rand(100, 2) 
estimator = KMeans(n_clusters=3)
estimator.fit(data)
label_pred = estimator.labels_ 
centroids = estimator.cluster_centers_ 
inertia = estimator.inertia_ 
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
color = 0
j = 0 
for i in label_pred:
    plt.plot([data[j:j+1,0]], [data[j:j+1,1]], mark[i], markersize = 5)
    j +=1
plt.show()

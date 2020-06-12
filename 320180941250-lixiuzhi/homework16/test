#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import KMeans

def create_data_set(*centers_old):#用于生成测试数据
    X = list()
    for x0, y0, z0 in centers_old:
        x = np.random.normal(x0, 0.1+np.random.random()/3, z0)
        y = np.random.normal(y0, 0.1+np.random.random()/3, z0)
        X.append(np.stack((x,y), axis=1))
    return np.vstack(X)


k = 4
X = create_data_set((0,0,2500), (0,2,2500), (2,0,2500), (2,2,2500))
print(X)

clf = KMeans(X)

plt.scatter(X[:,0], X[:,1])
plt.show() 


# In[ ]:





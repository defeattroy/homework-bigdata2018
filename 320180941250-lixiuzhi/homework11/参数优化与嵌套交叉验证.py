
# coding: utf-8

# In[10]:

from sklearn.datasets import load_iris  # 自带的样本数据集
from sklearn.neighbors import KNeighborsClassifier  # 要估计的是knn里面的参数，包括k的取值和样本权重分布方式 
from sklearn.model_selection import RandomizedSearchCV  # 网格搜索和随机搜索
 
iris = load_iris()
 
X = iris.data  
y = iris.target 

k_range = range(1, 100)  # 优化参数k的取值范围
weight_options = ['uniform', 'distance']  # 代估参数权重的取值范围。uniform为统一取权值，distance表示距离倒数取权值
# 下面是构建parameter grid，其结构是key为参数名称，value是待搜索的数值列表的一个字典结构
params = {'n_neighbors':k_range,'weights':weight_options}  # 定义优化参数字典，字典中的key值必须是分类算法的函数的参数名
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=18, p=2,
           weights='uniform')  # 定义分类算法。n_neighbors和weights的参数名称和params字典中的key名对应


rand = RandomizedSearchCV(knn, params, cv=10, scoring='accuracy', n_iter=10, random_state=5)  #可快速确定一个参数的大概范围
rand.fit(X, y)#调用fit进行网格搜索
 
#print('随机搜索-度量记录：',grid.cv_results_)  # 包含每次训练的相关信息
print('随机搜索-最佳度量值:',rand.best_score_)  # 获取最佳度量值
print('随机搜索-最佳参数：',rand.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
print('随机搜索-最佳模型：',rand.best_estimator_)  # 获取最佳度量时的分类器模型


# 使用获取的最佳参数生成模型，预测数据
knn_random = KNeighborsClassifier(n_neighbors=rand.best_params_['n_neighbors'], weights=rand.best_params_['weights'])  # 取出最佳参数进行建模
knn_random.fit(X, y)  # 训练模型
print(knn_random.predict([[3, 5, 4, 2]]))  # 预测新对象


# In[12]:

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


#获取数据集
iris=datasets.load_iris()
X=iris.data
y=iris.target
#通过train_test_spli将数据分为训练集、测试集，测试集占0.3的比例
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y,random_state = 1)
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state = 1))

#通过网格搜索优化 svc 模型
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000]
param_grid = [{'svc__C':param_range,'svc__kernel':['linear']},
            {'svc__C':param_range,'svc__kernel':['rbf'],
             'svc__gamma':param_range}]#对模型的3个参数搜索最优参数

gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid, scoring = 'accuracy', cv=2) #自动搜索，自动调参

scores = cross_val_score(gs,X_train,y_train,scoring = 'accuracy',cv = 5)#计算模型得分
print("CV Accuracy in Train Phase: %.3f +/- %.3f" % (np.mean(scores),np.std(scores)))

gs.fit(X_train,y_train)#调用fit进行网格搜索
print("Accuracy in Train Phase: %.3f" % gs.best_score_)


# In[ ]:




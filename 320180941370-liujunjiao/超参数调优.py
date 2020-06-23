'''
超参数调优
参考网址“https://blog.csdn.net/Amy_mm/article/details/79902477”
网格搜索
主要思想：
暴风搜索法，首先为不同的超参数设定一个值列表，然后计算机会遍历每个超参数的组合进行性能评估，选出性能最佳的参数组合。
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#引入鸢尾花数据
iris = datasets.load_iris()
X = iris.data#特征数据
y = iris.target#分类结果
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=1)


pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state = 1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
 
param_grid = [{'svc__C' : param_range,
               'svc__kernel' : ['linear']},
              {'svc__C' :  param_range,
               'svc__kernel':  ['rbf'],
               'svc__gamma' : param_range}]


gs=GridSearchCV(estimator=pipe_svc,
                param_grid=param_grid,
                scoring='accuracy',
                cv=2)

scores=cross_val_score(gs,X_train,y_train,scoring='accuracy',cv=5)
print("CV Accuracy in Train Phase: %.3f +/- %.3f" % (np.mean(scores),np.std(scores)))

gs.fit(X_train,y_train)
#输出最优超参数及其分数
print(gs.best_score_)
print(gs.best_params_)

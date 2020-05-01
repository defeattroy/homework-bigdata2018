#参数寻优——对参数列表进行穷举搜索，评估每个组合性能

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


#获取数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)


#用make_pipeline创建管道
pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))


#param_grid为调优参数列表
param_range=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000]
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},
            {'svc__C':param_range,'svc__kernel':['rbf'],
             'svc__gamma':param_range}]


#初始化GridSearchCV网格搜索对象，对svm流水线的训练和调优
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



#得到svm模型在超参数svc_C=10.0的时候可得到最优k折交叉验证准确率96.7%
#用最优参数进行模型性能评估
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))

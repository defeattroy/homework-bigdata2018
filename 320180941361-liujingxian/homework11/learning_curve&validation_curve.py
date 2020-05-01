from sklearn.model_selection import learning_curve #加载学习曲线
from sklearn.model_selection import validation_curve #加载验证曲线
from sklearn.model_selection import ShuffleSplit #加载数据处理
 
from sklearn import datasets  #加载数据包
from sklearn.naive_bayes import GaussianNB#加载高斯贝叶斯
from sklearn.svm import LinearSVC #加载支持向量机
 
import numpy as np
import matplotlib.pyplot as plt
 
data=datasets.load_iris()
x=data.data
y=data.target
 
cv=ShuffleSplit(test_size=0.25,random_state=0,n_splits=4) #处理数据，测试数据比例为0.25，4——折交叉
estimator=GaussianNB()
estimator.get_params().keys()  #可以获得学习算法参数

train_sizes=[0.1,0.2,0.5,0.7,1]
train_size,train_scores,test_scores=learning_curve(estimator,x,y,cv=cv,\
                                    train_sizes=[0.1,0.2,0.5,0.7,1]) #获得学习曲线，针对不同的数据集
 
new_train_scores=train_scores.mean(1)
train_std=train_scores.std()
test_std=test_scores.std()
new_test_scores=test_scores.mean(1)
'''
画出不同比例数据集的学习曲线
'''
plt.grid()
plt.fill_between(train_sizes,new_train_scores-train_std,
                 new_train_scores+train_std,color='r',alpha=0.1)
plt.fill_between(train_sizes,new_test_scores-test_std,
                 new_test_scores+test_std,color='g',alpha=0.1)
 
plt.plot(train_sizes,new_train_scores,'*-',c='r',label='train score')
plt.plot(train_sizes,new_test_scores,'*-',c='g',label='test score')
plt.legend(loc='best')
plt.show()

 

 
'''
使用支持向量机，来做验证曲线
'''
estimator2=LinearSVC()
estimator2.get_params().keys()#查看有哪些系数
train_score2,validation_score2=validation_curve(estimator2,x,y,param_name='C',cv=cv
                ,param_range=np.linspace(25,200,8)) #改变变量C，来看得分
 
x_axis=np.linspace(25,200,8)
train_score2_mean=train_score2.mean(1)
train_score2_std=train_score2.std(1)
validation_score2_mean=validation_score2.mean(1)
validation_score2_std=validation_score2.std(1)
 
plt.grid()
plt.fill_between(x_axis,train_score2_mean-train_score2_std,
                 train_score2_mean+train_score2_std,color='r',alpha=0.1)
plt.fill_between(x_axis,validation_score2_mean-validation_score2_std,
                 validation_score2_mean+validation_score2_std,color='g',alpha=0.1)
 
plt.plot(x_axis,train_score2_mean,'o-',c='r',label='train score')
plt.plot(x_axis,validation_score2_mean,'o-',c='g',label='validation score')
plt.legend(loc='best')
plt.show()


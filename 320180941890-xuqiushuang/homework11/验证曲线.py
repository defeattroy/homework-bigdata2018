#验证曲线
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,learning_curve,validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()
X = iris.data
y = iris.target
#分割训练集与测试集
sc=StandardScaler()
sc.fit(X)
X_std=sc.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.4,stratify=y,random_state=0)
#penalty为正则化选择参数，默认l2正则化
pipe_lr=Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(random_state=1,penalty='l2'))])
#统计结果
param_range=[0.001,0.01,0.1,1.0,10.0,100.0]
#10折，验证正则化参数C,cv:k-fold,param_name:将被改变的参数的名字，param_range:参数的改变范围
train_scores,test_scores =validation_curve(estimator=pipe_lr,X=X_train,y=y_train,param_name='clf__C',param_range=param_range,cv=10)
#统计结果
train_mean= np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean =np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
#绘制
plt.figure(figsize=(20,10))
plt.plot(param_range,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(param_range,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='test accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.6,1.3])
plt.show()

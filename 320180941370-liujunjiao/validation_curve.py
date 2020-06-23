'''
validation_curve曲线绘制
参考网址：
“https://blog.csdn.net/haha456487/article/details/103987011”
“https://blog.csdn.net/aliceyangxi1987/article/details/73621144”
（使用SVM模型）

'''
from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,validation_curve

#引入鸢尾花数据
iris = datasets.load_iris()
X = iris.data#特征数据
y = iris.target#分类结果
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

clf = svm.SVC()
param_range = np.logspace(-6, -1, 5)
train_scores, valid_scores = validation_curve( estimator=clf, X=x_train, y=y_train, param_name="gamma", param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)


# 可视化输出

plt.plot(param_range, train_mean,label='training scores')  
plt.plot(param_range, valid_mean,label='validation scores')    
plt.legend()

plt.show()
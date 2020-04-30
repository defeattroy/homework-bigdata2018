from sklearn import datasets
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np


#下载数据，分割数据
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3,random_state=1)

#标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

#计算数据
pip = Pipeline([('scl', StandardScaler()),('clf',(KNeighborsClassifier()))])
param_range = [1,2,3,4,5]
train_scores, valid_scores = validation_curve(pip,x_train,y_train,param_name='clf__n_neighbors',param_range=param_range,cv=10)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

#画图
plt.figure()
plt.plot(param_range, train_scores_mean, marker='o', markersize=5,label='training accuracy')
plt.fill_between(param_range,train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,alpha=0.1)
plt.plot(param_range, valid_scores_mean, marker='x', markersize=5,label='validation accuracy')
plt.fill_between(param_range,valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std,alpha=0.1)
plt.grid()
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.ylim(0.5,1.2)
plt.title("Validation Curve")
plt.legend()
plt.savefig('./curves/validation_curve.jpg')
plt.show

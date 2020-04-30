from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve


#引入数据集并分割数据集
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2,random_state=1)

#对特征值进行标准化处理
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

#计算数据
pipe_lr = Pipeline([('scl', StandardScaler()),('clf',(KNeighborsClassifier()))])
param_range = [3,4,5,7,8,9,10]
train_scores, valid_scores = validation_curve(estimator=pipe_lr,X=x_train,y=y_train,
    param_name='clf__n_neighbors',param_range=param_range,cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

#绘图
plt.plot(param_range, train_mean, c='blue', marker='o', markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean - train_std,train_mean + train_std,alpha=0.15, color='blue')
plt.plot(param_range, valid_mean, c='green', marker='o', markersize=5,label='validation accuracy')
plt.fill_between(param_range,valid_mean - valid_std,valid_mean + valid_std,alpha=0.15, color='green')
plt.grid()
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title("Validation_Curve")
plt.ylim([0.8, 1.0])
plt.savefig('./图片结果/Validation_Curve.jpg')

from sklearn import datasets
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

#下载数据，分割数据
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

#标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

#计算数据
pip = Pipeline([('scl', StandardScaler()),('clf',(KNeighborsClassifier(n_neighbors=5)))])
train_sizes, train_scores, valid_scores = learning_curve(pip, x_train, y_train, n_jobs=1, train_sizes=np.linspace(0.1,1,10), cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

#画图
plt.figure()
plt.plot(train_sizes, train_scores_mean, marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes,train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,alpha=0.1)
plt.plot(train_sizes, valid_scores_mean, marker='x', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes,valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std, alpha=0.1)
plt.grid()
plt.xlabel('Training samples')
plt.ylabel('Accuracy')
plt.title("Learning Curve")
plt.ylim([0.2, 1.1])
plt.legend()
plt.savefig('./curves/learning_curve.jpg')
plt.show()

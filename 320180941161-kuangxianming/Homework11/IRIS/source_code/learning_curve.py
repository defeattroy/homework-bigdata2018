from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve


def get_iris_data():
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    return iris_data, iris_target

 # 1.获取鸢尾花的特征值，目标值
iris_data, iris_target = get_iris_data()
# 2.将数据分割成训练集和测试集 test_size=0.25表示将25%的数据用作测试集
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25,random_state=0)
# 3.特征工程(对特征值进行标准化处理)
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('clf',(KNeighborsClassifier(n_neighbors=5)))])

train_sizes, train_scores, valid_scores = learning_curve( estimator=pipe_lr, X=x_train, y=y_train,
                                                          train_sizes=np.linspace(0.1,1,10), cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.plot(train_sizes, train_mean, c='blue', marker='o', markersize=5,
         label='training accuracy')
plt.fill_between(train_sizes,
                 train_mean - train_std,
                 train_mean + train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, valid_mean, c='green', marker='o', markersize=5,
         linestyle='--', label='validation accuracy')
plt.fill_between(train_sizes,
                 valid_mean - valid_std,
                 valid_mean + valid_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.ylim([0.5, 1])
plt.show()

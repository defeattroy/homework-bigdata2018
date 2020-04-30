from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#下载数据，分割数据
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

#标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

#参数寻优
params = {'n_neighbors':[i for i in range(1,11)]}
knn_clf = KNeighborsClassifier()
gs = GridSearchCV(knn_clf, params)
gs.fit(x_train, y_train)
print('准确率：',gs.score(x_test, y_test))
print('最佳分类器：',gs.best_estimator_)
print('最佳分类器对应的准确度：',gs.best_score_)
print('最佳分类器对应的参数：',gs.best_params_)

#交叉验证
knn_clf = gs.best_estimator_
knn_clf.fit(x_train, y_train)
scores = cross_val_score(knn_clf, iris.data, iris.target, cv=10)
print('10折交叉验证后的得分：',scores)
print('平均得分：',scores.mean())

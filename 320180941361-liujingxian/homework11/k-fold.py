from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

knn_test = KNeighborsClassifier()
params = {"n_neighbors": [3, 4, 8, 10]}
gridCv = GridSearchCV(knn_test, param_grid=params, cv=5)
gridCv.fit(X_train, y_train)  
print("k-flod交叉验证中最好的结果：", gridCv.best_score_)
print("最好的模型参数是：", gridCv.best_estimator_.n_neighbors)
k_neighbor=gridCv.best_estimator_.n_neighbors

std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=k_neighbor)  
knn.fit(X_train, y_train)  
y_predict = knn.predict(X_test)

labels = ["山鸢尾", "虹膜锦葵", "变色鸢尾"]
tplt = "{0:{3}^10}\t{1:{3}^10}\t{2:^10}"
print(tplt.format("第i次测试","真实值","预测值",chr(12288)))
for i in range(len(y_predict)):
    print(tplt.format((i+1),labels[y_predict[i]],labels[y_test[i]],chr(12288)))
print("准确率为",knn.score(X_test, y_test))

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


#获取数据并将数据分割，将25%的数据用作测试集
iris = datasets.load_iris()
iris_data = iris.data
iris_target = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=1)
#利用KNN算法建模，参数优化，寻找最优模型参数
knn_test = KNeighborsClassifier()
params = {"n_neighbors": [3, 5, 7, 9, 10]}
gridCv = GridSearchCV(knn_test, param_grid=params, cv=5)
gridCv.fit(x_train, y_train)  
print("交叉验证中最好的结果：", gridCv.best_score_)
print("最好的模型参数是：", gridCv.best_estimator_.n_neighbors)
k_neighbor=gridCv.best_estimator_.n_neighbors
#对特征值进行标准化处理
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
#建模并进行预测
knn = KNeighborsClassifier(n_neighbors=k_neighbor)  
knn.fit(x_train, y_train)  
y_predict = knn.predict(x_test)  
#展示预测结果
labels = ["山鸢尾", "虹膜锦葵", "变色鸢尾"]
for i in range(len(y_predict)):
	print("第%d次测试:真实值:%s\t预测值:%s"%((i+1),labels[y_predict[i]],labels[y_test[i]]))
print("准确率：",knn.score(x_test, y_test))

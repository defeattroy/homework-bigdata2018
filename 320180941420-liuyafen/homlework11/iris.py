from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

"""
下载鸢尾花数据
萼片长度、萼片宽度、花瓣长度、花瓣宽度
"""
iris = load_iris()

"""
data对应了样本的4个特征，150行4列
target对应了样本的类别（目标属性），150行1列
iris.target用0、1和2三个整数分别代表了花的三个品种
"""
X = iris.data
y = iris.target



"""
选择总数据的30％的数据
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

"""
用KNN算法建模，参数优化，寻找最优模型参数
"""
knn_test = KNeighborsClassifier()
params = {"n_neighbors": [3, 4, 8, 10]}
gridCv = GridSearchCV(knn_test, param_grid=params, cv=5)
gridCv.fit(X_train, y_train)  
print("k-flod交叉验证中最好的结果：", gridCv.best_score_)
print("最好的模型参数是：", gridCv.best_estimator_.n_neighbors)
k_neighbor=gridCv.best_estimator_.n_neighbors

"""
对特征值进行标准化处理
"""
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

"""
建模并进行预测
"""
knn = KNeighborsClassifier(n_neighbors=k_neighbor)  
knn.fit(X_train, y_train)  
y_predict = knn.predict(X_test)

"""结果展示"""
labels = ["山鸢尾", "虹膜锦葵", "变色鸢尾"]
tplt = "{0:{3}^10}\t{1:{3}^10}\t{2:^10}"
print(tplt.format("第i次测试","真实值","预测值",chr(12288)))
for i in range(len(y_predict)):
    print(tplt.format((i+1),labels[y_predict[i]],labels[y_test[i]],chr(12288)))
print("准确率为",knn.score(X_test, y_test))

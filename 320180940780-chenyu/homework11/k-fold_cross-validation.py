from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
# K折交叉验证模块
from sklearn.model_selection import cross_val_score 
# 导入k聚类算法
from sklearn.neighbors import KNeighborsClassifier 

# 加载iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
# 建立Knn模型
knn = KNeighborsClassifier()
# 使用K折交叉验证模块
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
# 将10次的预测准确率打印出
print(scores)
print(scores.mean())

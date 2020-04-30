#KNN分类iris，交叉验证，参数选择并可视化

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
# 导入交叉验证
from sklearn.model_selection import cross_val_score

# 从datasets中导入iris数据，包含150条样本，每条样本4个feature
iris_data = datasets.load_iris()
iris_x = iris_data.data
iris_y = iris_data.target

# 尝试n_neighbors为不同值时，模型准确度
nb = range(1, 31)
# 保存每次n_neighbors对应准确率，用于plt画图
k_scores = []
for k in nb:
    # 使用KNN模型
    knn = KNeighborsClassifier(n_neighbors=k)
    # 使用交叉验证，不需要自己去切分数据集，也不需要knn.fit()和knn.predict(),cv=5表示交叉验证5组
    scores = cross_val_score(knn, iris_x, iris_y, cv=5, scoring='accuracy')
    # 取交叉验证集的平均值
    k_scores.append(scores.mean())

# 画出n_neighbor于accuracy的关系图
plt.plot(nb,k_scores)
plt.xlabel("Value of n_neighbors")
plt.ylabel("Value of Accuracy")
plt.show()



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


class KNN(object):
    """

    利用KNN算法对鸢尾花进行分类
    """

    # 获取鸢尾花数据 三个类别(山鸢尾/0，虹膜锦葵/1，变色鸢尾/2)，每个类别50个样本，每个样本四个特征值(萼片长度，萼片宽度
    # ，花瓣长度，花瓣宽度)

    def get_iris_data(self):
        iris = load_iris()
        iris_data = iris.data
        iris_target = iris.target

        return iris_data, iris_target

    def run(self):
        # 1.获取鸢尾花的特征值，目标值
        iris_data, iris_target = self.get_iris_data()
        # 2.将数据分割成训练集和测试集 test_size=0.25表示将25%的数据用作测试集
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25, random_state=0)
        # 3.特征工程(对特征值进行标准化处理)
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)

        # 4.送入算法
        knn = KNeighborsClassifier(n_neighbors=5)  # 创建一个KNN算法实例，n_neighbors默认为5,后续通过网格搜索获取最优参数
        knn.fit(x_train, y_train)  # 将测试集送入算法
        y_predict = knn.predict(x_test)  # 获取预测结果
        # 预测结果展示
        labels = ["山鸢尾", "虹膜锦葵", "变色鸢尾"]
        for i in range(len(y_predict)):
            print("第%d次测试:真实值:%s\t预测值:%s" % ((i + 1), labels[y_predict[i]], labels[y_test[i]]))
        print("准确率：", knn.score(x_test, y_test))
        scores = cross_val_score(knn, iris_data, iris_target, cv=5)
        print('5折交叉验证后的得分：'+scores)
        print('5折交叉验证平均得分为：'+scores.mean())


    def parameter_optimization(self):
        iris_data, iris_target = self.get_iris_data()
        # 2.将数据分割成训练集和测试集 test_size=0.25表示将25%的数据用作测试集
        x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25, random_state=0)
        # 生成knn估计器
        knn = KNeighborsClassifier()
        # 构造超参数值
        params = {"n_neighbors": [3, 5, 10]}
        # 进行网格搜索
        gridCv = GridSearchCV(knn, param_grid=params, cv=5)
        gridCv.fit(x_train, y_train)  # 输入训练数据
        # 预测准确率
        print("准确率：", gridCv.score(x_test, y_test))
        print("交叉验证中最好的结果：", gridCv.best_score_)
        print("最好的模型：", gridCv.best_estimator_)


if __name__ == '__main__':
    knn = KNN()
    knn.run()
    knn.parameter_optimization()
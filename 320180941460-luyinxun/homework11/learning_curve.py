#学习曲线——诊断模型的偏差和方差
#不同训练集下的准确率

from sklearn.model_selection import learning_curve,ShuffleSplit
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


#获取数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

#验证分离策略，交叉验证
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(C = 1100)
#train_sizes表示训练集是等间距间隔的10个样本
train_sizes,train_scores,test_scores = learning_curve(estimator = estimator, X = X, y = y, 
                                                      cv = cv, train_sizes = np.linspace(.1, 1.0, 10))


train_scores_mean = np.mean(train_scores,axis = 1)
train_scores_std = np.std(train_scores, axis = 1)
test_scores_mean = np.mean(test_scores, axis = 1)
test_scores_std = np.std(test_scores, axis = 1)


print(train_sizes)
plt.plot(train_sizes, train_scores_mean, 'o-', label = 'train score')
plt.plot(train_sizes, test_scores_mean, 'o-', label = 'cross-validation score')
plt.fill_between(train_sizes,train_scores_mean + train_scores_std,
                 train_scores_mean - train_scores_std,alpha=0.1)
plt.fill_between(train_sizes,test_scores_mean + test_scores_std,
                 test_scores_mean - test_scores_std,alpha=0.1)


plt.legend(loc='lower right')
plt.grid()
plt.title('Learning_curve of iris with svm')
plt.xlabel('Number of training sample')
plt.ylabel('Score')
plt.show()

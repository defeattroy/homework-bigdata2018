#验证曲线——诊断曲线的欠拟合和过拟合
#不同模型参数值下的准确率

from sklearn.model_selection import validation_curve,ShuffleSplit
from sklearn import datasets
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


#获取数据
iris = datasets.load_iris()
X = iris.data
y = iris.target


#设置验证训练比例，进行交叉验证
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC()
#超参数设为C，param_range设置C区间,寻找最优值
train_scores, validation_scores = validation_curve(estimator = estimator, X = X, y = y, param_name='C',
                                                  cv = cv, param_range = np.linspace(0.1,2000,10))

            
train_scores_mean = np.mean(train_scores,axis = 1)
train_scores_std = np.std(train_scores, axis = 1)
validation_scores_mean = np.mean(validation_scores, axis = 1)
validation_scores_std = np.std(validation_scores, axis = 1)
x_axis = np.linspace(0.1,2000,10)

  
plt.plot(x_axis, train_scores_mean, 'o-', label = 'train score')
plt.plot(x_axis, validation_scores_mean, 'o-', label = 'cross-validation score')
plt.fill_between(x_axis,train_scores_mean + train_scores_std,
                 train_scores_mean - train_scores_std,alpha=0.1)
plt.fill_between(x_axis,validation_scores_mean + validation_scores_std,
                 validation_scores_mean - validation_scores_std,alpha=0.1)


plt.legend(loc ='lower right')
plt.grid()
plt.title('Validation_curve of iris with SVM')
plt.xlabel('parameter C')
plt.ylabel('Score')
plt.show()

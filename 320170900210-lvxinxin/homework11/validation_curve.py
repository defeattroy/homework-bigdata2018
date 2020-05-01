#使用交叉验证，并画出验证曲线validation_curve，用于观察模型参数不同时的准确率

import numpy as np
import matplotlib.pyplot as plt

# 导入sklearn提供的验证曲线
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

# 导入数据
digits = load_digits()
X = digits.data
y = digits.target

# SVC参数gamma的范围
param_range = np.logspace(-6, -2.3, 5)

# 使用validation曲线，指定params的名字和范围
train_loss, test_loss = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=param_range, cv=10, scoring='neg_mean_squared_error'
)

# 将每次训练集交叉验证（10个损失值，因为cv=10）取平均值
train_loss_mean = -np.mean(train_loss, axis=1)
print(train_loss_mean)
# 将每次测试集交叉验证取平均值
test_loss_mean = -np.mean(test_loss, axis=1)
print(test_loss_mean)
# 画图，红色是训练平均损失值，绿色是测试平均损失值,注意这里的x坐标是param_range
plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(param_range, test_loss_mean, 'o-', color='g', label='Cross_validation')
plt.xlabel('Gamma')
plt.ylabel('Loss')
plt.show()

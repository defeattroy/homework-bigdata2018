#使用交叉验证，并画出学习曲线learning_curve，用于观察模型拟合情况

import numpy as np
import matplotlib.pyplot as plt

# 导入sklearn提供的损失曲线
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

# 导入数据
digits = load_digits()
X = digits.data
y = digits.target

# 使用学习曲线获取每个阶段的训练损失和交叉测试损失,train_sizes表示各个不同阶段，从10%到100%
train_sizes, train_loss, test_loss = learning_curve(
    SVC(gamma=0.001), X, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1, 10)
)

# 将每次训练集交叉验证（10个损失值，因为cv=10）取平均值
train_loss_mean = -np.mean(train_loss, axis=1)
print(train_loss_mean)
# 将每次测试集交叉验证取平均值
test_loss_mean = -np.mean(test_loss, axis=1)
print(test_loss_mean)
# 画图，红色是训练平均损失值，绿色是测试平均损失值
plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Cross_validation')
plt.xlabel('Train sizes')
plt.ylabel('Loss')
plt.show()


# coding: utf-8

# In[6]:

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

digits = load_digits()
X, y = digits.data, digits.target

#建立参数测试集，从1e-6到1e-2次方，分五段
param_range = np.logspace(-6, -2, 5)#设置值的范围

#使用validation_curve快速找出参数对模型的影响
train_scores, test_scores = validation_curve(SVC(), X, y, param_name="gamma", param_range = param_range, cv = 10, scoring = "accuracy", n_jobs = 1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
#半对数坐标函数：只有一个坐标轴是对数坐标，另一个是普通算术坐标
plt.semilogx(param_range, train_scores_mean, label = "Training score", color = "r", lw = lw)
#在区域内绘制函数包围的区域
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.2, color = "darkorange", lw = lw) 
#通过fill_between函数假如平均准确率标准差的信息，表示评估结果的方差
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color = "g", lw = lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.2, color = "navy", lw = lw)
plt.legend(loc = "best")
plt.show()


# In[ ]:




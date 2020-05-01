
# coding: utf-8

# In[3]:

###朴素贝叶斯分类
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

iris = datasets.load_iris() # 加载鸢尾花数据
iris_x = iris.data  # 获取数据
# print(iris_x)
iris_x = iris_x[:, :2]  # 取前两个特征值
# print(iris_x)
iris_y = iris.target    # 0， 1， 2
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.75, random_state=1) # 对数据进行分类 一部分最为训练一部分作为测试
# clf = GaussianNB()
# ir = clf.fit(x_train, y_train)
clf = Pipeline([
         ('sc', StandardScaler()),
         ('clf', GaussianNB())])     # 管道这个没深入理解 所以不知所以然
ir = clf.fit(x_train, y_train.ravel())  # 利用训练数据进行拟合
 
# 画图：   
x1_max, x1_min = max(x_test[:, 0]), min(x_test[:, 0])   # 取0列特征得最大最小值
x2_max, x2_min = max(x_test[:, 1]), min(x_test[:, 1])   # 取1列特征得最大最小值
t1 = np.linspace(x1_min, x1_max, 500)   # 生成500个测试点
t2 = np.linspace(x2_min, x2_max, 500)   
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_test1 = np.stack((x1.flat, x2.flat), axis=1)
y_hat = ir.predict(x_test1) # 预测
mpl.rcParams['font.sans-serif'] = [u'simHei']   # 识别中文保证不乱吗
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF']) # 测试分类的颜色
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])    # 样本点的颜色
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_hat.reshape(x1.shape), cmap=cm_light)  # y_hat  25000个样本点的画图，
plt.scatter(x_test[:, 0], x_test[:, 1], edgecolors='k', s=50, c=y_test, cmap=cm_dark)   # 测试数据的真实的样本点（散点） 参数自行百度
plt.xlabel(u'花萼长度', fontsize=14)
plt.ylabel(u'花萼宽度', fontsize=14)
plt.title(u'朴素贝叶斯对鸢尾花数据的分类', fontsize=18)
plt.grid(True)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()
y_hat1 = ir.predict(x_test)
result = y_hat1 == y_test
print(result)
acc = np.mean(result)
print('准确度: %.2f%%' % (100 * acc))


# In[6]:

### SVM分类器

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm  # sklearn自带SVM分类器
from sklearn import datasets 
from sklearn.model_selection import train_test_split
 
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
 

#获取鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 取数据集的特征向量
Y = iris.target  # 取数据集的标签（鸢尾花类型）
X = X[:, 0:2] # 取前两列特征向量，用来作二特征分类
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, random_state = 1)
 

#SVM 分类器
clf = svm.SVC(C = 0.8, kernel = 'rbf', gamma = 20, decision_function_shape = 'ovo')
clf.fit(x_train, y_train.ravel())
 
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()  # 第0列的范围
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()  # 第1列的范围
xx, yy = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
 
grid_test = np.stack((xx.flat, yy.flat), axis = 1)  # 测试点
grid_hat = clf.predict(grid_test) # 预测分类值
grid_hat = grid_hat.reshape(xx.shape) # 使之与输入的形状相同
 
plt.pcolormesh(xx, yy, grid_hat, cmap = cm_light)
plt.scatter(X[:, 0], X[:, 1], c = Y, edgecolors = 'k', s = 50, cmap = cm_dark)  # 样本
plt.scatter(x_test[:, 0], x_test[:, 1], s = 120, facecolors = 'none', zorder = 10)  # 圈中测试集样本
 
plt.xlabel('花萼长度', fontsize = 14)
plt.ylabel('花萼宽度', fontsize = 14)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('SVM对鸢尾花数据的分类', fontsize = 15)
plt.grid(b = True, ls = ':')
plt.show()
 

#计算准确率
print("准确率 %f" %(clf.score(x_train, y_train))) # 训练集准确率
print("准确率 %f" %(clf.score(x_test, y_test))) # 测试集准确率


# In[30]:

###决策树分类
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 训练模型，限制树的最大深度5
clf = DecisionTreeClassifier(max_depth=5)
#拟合模型
clf.fit(X, y)

# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# 绘制等高线plt.contourf() 区域颜色填充 alpha 指透明度
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()


# In[31]:

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 训练模型，限制树的最大深度2
clf = DecisionTreeClassifier(max_depth=2)
#拟合模型
clf.fit(X, y)

# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# 绘制等高线plt.contourf() 区域颜色填充 alpha 指透明度
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




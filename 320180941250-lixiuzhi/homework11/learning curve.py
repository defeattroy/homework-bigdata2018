
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#加载数据集
iris = datasets.load_iris()
data = iris.data
target = iris.target
#print(data.shape,  target.shape)


#分层采样划分训练集和测试集
split = ShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(data, target):
    X_train, y_train = data[train_index], target[train_index]
    X_test, y_test = data[test_index], target[test_index]
    
#print(X_train.shape, y_train.shape)

#设置超参数c=0.05，gamme=0.1训练SVM模型
svm_pip = Pipeline([("scaler", StandardScaler()), ("svm", SVC(C = 0.05, gamma = 0.1, kernel = 'rbf'))])
scores = cross_val_score(svm_pip, X_train, y_train, cv = 3, scoring = "accuracy")
#print("scores:{}, scores mean:{} +/- {}".format(scores, np.mean(scores), np.std(scores)))  #结果准确率为0.66

train_sizes, train_scores, test_scores = learning_curve(svm_pip, X_train, y_train, cv = 3, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5), scoring = "accuracy")
#train_size用于控制生产学习曲线的样本的绝对或者相对数量
train_scores_mean = np.mean(train_scores, axis=1)  #将训练得分集合按行得到平均值
train_scores_std = np.std(train_scores, axis=1)  #计算训练矩阵的标准方差
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()  #背景设置为网格线

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")   #函数画出模型准确性的平均值
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.title("Learning Curve with SVM")
plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()
#模型训练准确率和验证准确率同步变化，现处于欠拟合状态


# In[40]:

#设置超参数c=10，gamme=1，降低对模型的约束
svm_pip = Pipeline([("scaler", StandardScaler()), ("svm", SVC(C = 10, gamma = 1, kernel = 'rbf'))])
scores = cross_val_score(svm_pip, X_train, y_train, cv = 3, scoring = "accuracy")
#print("scores:{}, scores mean:{} +/- {}".format(scores, np.mean(scores), np.std(scores)))  #结果准确率为0.916

train_sizes, train_scores, test_scores = learning_curve(svm_pip, X_train, y_train, cv = 3, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5), scoring = "accuracy")
#train_size用于控制生产学习曲线的样本的绝对或者相对数量，通过cv设置k值
train_scores_mean = np.mean(train_scores, axis=1)  #将训练得分集合按行得到平均值
train_scores_std = np.std(train_scores, axis=1)  #计算训练矩阵的标准方差
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()  #背景设置为网格线

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")   #函数画出模型准确性的平均值
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.title("Learning Curve with SVM")
plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()
#模型随着训练集的增大，训练集上准确率基本是百分百，验证集准确率在百分之九十左右，模型在训练集和验证集上的相差结果较大，处于与过拟合状态


# In[41]:

#设置超参数c=5，gamme=0.05，增强对模型的约束
svm_pip = Pipeline([("scaler", StandardScaler()), ("svm", SVC(C = 5, gamma = 0.05, kernel = 'rbf'))])
scores = cross_val_score(svm_pip, X_train, y_train, cv = 3, scoring = "accuracy")
print("scores:{}, scores mean:{} +/- {}".format(scores, np.mean(scores), np.std(scores)))  #结果准确率为0.95

train_sizes, train_scores, test_scores = learning_curve(svm_pip, X_train, y_train, cv = 3, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5), scoring = "accuracy")
#train_size用于控制生产学习曲线的样本的绝对或者相对数量
train_scores_mean = np.mean(train_scores, axis=1)  #将训练得分集合按行得到平均值
train_scores_std = np.std(train_scores, axis=1)  #计算训练矩阵的标准方差
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()  #背景设置为网格线

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")   #函数画出模型准确性的平均值
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.title("Learning Curve with SVM")
plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc = "best")
plt.show()


# In[ ]:




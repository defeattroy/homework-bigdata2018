'''
K折交叉验证法
（使用决策树模型）
'''
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import numpy as np

#引入鸢尾花数据
iris = datasets.load_iris()
X = iris.data#特征数据
y = iris.target#分类结果

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
clf=DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

print("十次准确率如下：")
for i in scores:
    print(i)
print("平均值为：")
print(scores.mean())
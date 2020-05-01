#k折交叉验证

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,learning_curve,validation_curve,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()
X = iris.data
y = iris.target
#分割训练集与测试集
sc=StandardScaler()
sc.fit(X)
X_std=sc.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.4,stratify=y,random_state=0)
kfold = StratifiedKFold(n_splits=10).split(X,y)
scores = []
lr = LogisticRegression(C=1, random_state=0)#C：正则化系数的倒数
for k, (train,test) in enumerate(kfold):
    lr.fit(X_train, y_train)
    score = lr.score(X_test,y_test)
    scores.append(score)
    print("Fold %d, Accuracy: %.3f" % (k+1, score))
    
print("Mean: %.3f, +- %.3f" % (np.mean(scores), np.var(scores)))

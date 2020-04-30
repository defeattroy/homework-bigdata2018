#参数寻优
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,learning_curve,validation_curve,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data
y = iris.target
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

X_train_std, X_test_std, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=1,stratify=y)

param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
param_grid = [{'C': param_range, 'kernel': ['linear']},
            {'C': param_range, 'kernel': ['rbf'], 'gamma':param_range}]

gs = GridSearchCV(estimator=SVC(random_state=1),
                 param_grid = param_grid,
                 scoring = 'accuracy',
                 cv=10)
# 训练并找到最好的参数
gs = gs.fit(X_train_std, y_train)
print(gs)
print(gs.best_score_)
print(gs.best_params_)

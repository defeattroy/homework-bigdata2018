"""
@Time ： 2020/5/1 0:25
@Auth ： Erris
@Version:  Python 3.8.0

"""

import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

iris = load_iris()

#learning curve
K = 5

X = iris.data[:,:2]
y = iris.target

sc = StandardScaler()
sFolder = StratifiedKFold(n_splits=K, shuffle=True).split(X,y)

it = 0
learn_score = [[] for i in range(K)]
test_score = [[] for i in range(K)]


for k, (train,test)in enumerate(sFolder):
    sc.fit(X[train])
    X_train_std = sc.transform(X[train])
    X_test_std = sc.transform(X[test])
    for i in range(60, 121, 4):
        clf = svm.SVC(C=0.8, kernel='rbf', gamma=1, decision_function_shape='ovr')
        clf.fit(X_train_std[:i], y[train][:i])
        
        learn_score[it].append(clf.score(X_train_std[:i], y[train][:i]))
        test_score[it].append(clf.score(X_test_std, y[test]))
    it+=1
    
learn = np.array([0 for i in range(len(learn_score[0]))])
test = np.array([[0 for i in range(len(learn_score[0]))]])
for i in range(K):
    ls = np.array(learn_score[i])
    ts = np.array(test_score[i])
    learn = learn+ls
    test = test+ts
    
learn_score = learn/K
test_score = test/K
        
    
plt.plot([i for i in range(60, 121, 4)], learn_score, label='train')
plt.plot([i for i in range(60, 121, 4)], test_score[0], label='test')
                    
plt.legend()
plt.show()

#In this case, the number of data does less than gamma
#validation curve
K = 6

sc = StandardScaler()

X = iris.data[:,:2]
y = iris.target

it = 0
val_gamma = [i/10 for i in range(1, 100)] #gamma from 0.1 to 10.0
avg_score = [0 for i in val_gamma]

for val in val_gamma:
    sFolder = StratifiedKFold(n_splits=K, random_state=0, shuffle=True).split(X,y)
    for k, (train,test)in enumerate(sFolder):
        sc.fit(X[train])
        X_train_std = sc.transform(X[train])
        X_test_std = sc.transform(X[test])
        clf = svm.SVC(C=0.8, kernel='rbf', gamma=val, decision_function_shape='ovr')
        clf.fit(X_train_std, y[train])
        clf.fit(X_train_std, y[train])
        avg_score[it] += clf.score(X_test_std, y[test])
    avg_score[it]/=K
    it+=1
    
plt.plot(val_gamma, avg_score)
plt.show()


#ROC curve
sc = StandardScaler()
sFolder = StratifiedKFold(n_splits=K, random_state=0, shuffle=True).split(X,y)

X = iris.data[:,:2]
y = iris.target

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
clf = OneVsRestClassifier(svm.SVC(C=0.8, kernel='rbf', gamma=1, decision_function_shape='ovr'))
scores = clf.fit(X_train_std, y_train).decision_function(X_test_std)
    
fpr = tpr = []
    
for i in range(n_classes):
    fpr_t, tpr_t, _ = roc_curve(y_test[:, i], scores[:, i])
    plt.plot(fpr_t, tpr_t)
plt.show()
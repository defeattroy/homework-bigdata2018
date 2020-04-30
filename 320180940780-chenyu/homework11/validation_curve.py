from sklearn import datasets,svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,validation_curve

iris = datasets.load_iris()
X = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

model = svm.SVC()
param_range = np.logspace(-6, -1, 5)
train_scores, valid_scores = validation_curve( estimator=model, X=x_train, y=y_train, param_name="gamma", param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)


# 可视化输出

plt.plot(param_range, train_mean, color='#ff0000', label='training scores')  
plt.plot(param_range, valid_mean, color='#39c5bb', label='validation scores')    
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()

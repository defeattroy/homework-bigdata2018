from sklearn import datasets,svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,learning_curve

iris = datasets.load_iris()
X = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

model = svm.SVC(kernel='linear')

train_sizes, train_scores, valid_scores = learning_curve( estimator=model, X=x_train, y=y_train,
                                                          train_sizes=np.linspace(0.1,1,10), cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)


# 可视化输出

plt.plot(train_sizes, train_mean, color='#66ccff', label='training accuracy')  
plt.plot(train_sizes, valid_mean, color='#ff0000', label='validation accuracy')    
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.show()

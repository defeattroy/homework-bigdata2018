import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import svm

# Purpose : Draw a validation curve, to view the effect of hyperparameter gamma on svm modeling

# Load data to pandas' Dataframe.
iris_data = load_iris()
target = pd.DataFrame(data=iris_data.target)
data = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
# Split train and test data.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
# Normalization & Standardization
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# Modeling
classifier = svm.SVC(C=30, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr: one to many
param_range = range(1, 10)

# Caculate for validation curve , see the gamma's effect to moldle
train_scores, valid_scores = validation_curve(classifier, x_train, y_train.values.ravel(),
                                              param_name='gamma', param_range=param_range, cv=6, scoring="accuracy")
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)

plt.plot(param_range, train_mean, c='blue', marker='o', markersize=5,
         label='training accuracy')
# Because the accuracy is very high, this fill function is equivalent to no effect
plt.fill_between(param_range,
                 train_mean - train_std,
                 train_mean + train_std,
                 alpha=0.15, color='blue')

plt.plot(param_range, valid_mean, c='green', marker='o', markersize=5,
         label='validation accuracy')
plt.fill_between(param_range,
                 valid_mean - valid_std,
                 valid_mean + valid_std,
                 alpha=0.15, color='green', label=r'$\pm$ train_std')

plt.grid()

plt.xlabel('Parameter gamma')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.ylim([0.8, 1.01])

plt.show()

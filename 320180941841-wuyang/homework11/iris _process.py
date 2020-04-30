from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import svm

import pandas as pd

# Purpose: Use svm Classifier and k-fold cross-validation. And parameter optimization with Nested cross validation(GridSearchCV).

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

classifier = svm.SVC(C=30, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr: one to many
classifier.fit(x_train,y_train.values.ravel())
c_score = cross_val_score(classifier, x_train, y_train.values.ravel(), cv=6)
print("6-flod score:" + str(c_score))
print("6-flod mean_socre:" + str(c_score.mean()))
print("Test_dataset score：", classifier.score(x_train, y_train))
print("Train_dataset score：", classifier.score(x_test, y_test))

# Parameter optimization
# method one : use GridSearchCV (A Nested cross validation method)
classifier = svm.SVC(kernel='rbf', decision_function_shape='ovr')  # Generate SVM
params = {"C": range(1, 10), "gamma": range(1, 11)}
gridCv = GridSearchCV(classifier, param_grid=params, cv=6)  # 6-fold nested cross-validation
gridCv.fit(x_train, y_train.values.ravel())

print("\nAccuracy ：", gridCv.score(x_test, y_test))
print("Best Cross score", gridCv.best_score_)
print("Best modle：", gridCv.best_estimator_)

# method two : Manual calculation
score = []
for C in range(1, 10):
    for gamma in range(1, 11):
        classifier = svm.SVC(C=C, kernel='rbf', gamma=gamma, decision_function_shape='ovr')  # ovr: One-to-many strategy
        classifier.fit(x_train, y_train.values.ravel())
        cross_score = cross_val_score(classifier, x_train, y_train.values.ravel(), cv=6)  # 6-fold cross-validation
        score.append((classifier.score(x_train, y_train), classifier.score(x_test, y_test)))
print("\nAccuracy ：", max([x[1] for x in score]))
print("Best Cross score", max([x.mean() for x in cross_score]))
# Next, to find the parameter corresponding to this maximum value, it is a bit more troublesome than the previous method
# And I found that the results of this method are not exactly the same, but they are similar.

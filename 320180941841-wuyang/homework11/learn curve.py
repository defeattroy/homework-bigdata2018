import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Purpose : Draw a learn curve.

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Draw a learn curve
    :param estimator: train method
    :param title: title of plot
    :param X: x_train
    :param y: y_train
    :param ylim: y axis
    :param cv: Number of cross-validated copies
    :param n_jobs: Optional number of jobs to run in parallel
    :param train_sizes:  x axis
    :return: a learn curve
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()

# load & translate & split data
data = load_iris()
target = pd.DataFrame(data=data.target)
data = pd.DataFrame(data=data.data, columns=data.feature_names)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

# Normalization & Standardization
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# we find the best param : C = 2 & gamma = 2
classifier = svm.SVC(C=2, kernel='rbf', gamma=2, decision_function_shape='ovr')  # ovr: one to many
classifier.fit(x_train, y_train.values.ravel())

title = r"Learning Curves for Iris dataset (SVM)"

plot_learning_curve(classifier, title, x_train, y_train.values.ravel(), ylim=(0.5, 1.01), cv=6, n_jobs=1,
                    train_sizes=np.linspace(0.1, 1, 10))

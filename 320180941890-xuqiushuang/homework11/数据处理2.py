import numpy as np
import doctest
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset=load_iris()

#花的品种
print("Target names:\n{}".format(iris_dataset['target_names']))
#特征
print("Feature names:\n{}".format(iris_dataset['feature_names']))
#数据集的规模
print("Shape of data:{}".format(iris_dataset['data'].shape))

'''
Target names:
['setosa' 'versicolor' 'virginica']
Feature names:
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Shape of data:(150, 4)
>>>

'''
if __name__=='__main__':
  doctest.testmod(verbose=True)

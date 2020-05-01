from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris_dataset=load_iris()
iris=datasets.load_iris()
X=iris.data
y=iris.target
sc=StandardScaler()
sc.fit(X)
X_std=sc.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_std,y,
test_size=0.4,stratify=y,random_state=0)

print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))
print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))
print("X_train:{}".format(X_train))
print("y_train:{}".format(y_train))

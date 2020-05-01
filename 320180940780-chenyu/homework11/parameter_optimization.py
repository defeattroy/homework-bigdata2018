from sklearn import svm,datasets
from sklearn.model_selection import train_test_split,GridSearchCV

iris = datasets.load_iris()
X = iris.data
y = iris.target

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

param_grid = {'kernel': ('rbf', 'linear'), 'C': [1, 5, 10]}
svr = svm.SVC(random_state=1,gamma='auto')
# 对所有的参数组合进行测试，选择最好的一个组合
clf = GridSearchCV(svr, param_grid)
clf = clf.fit(X_train, y_train)
print(clf)

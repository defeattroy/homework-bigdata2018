import pandas as pd
from sklearn import  datasets
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier

iris = datasets.load_iris()
parameters = {'n_estimators':range(100,150,10),'max_depth':range(3,5,1)}
xgc=XGBClassifier()
clf = RandomizedSearchCV(xgc, parameters,cv=5)
clf.fit(iris.data, iris.target)
cv_result = pd.DataFrame.from_dict(clf.cv_results_)

best_param=clf.best_params_
best_score=clf.best_score_

y_pred = clf.predict(iris.data)
print(classification_report(y_true=iris.target, y_pred=y_pred))

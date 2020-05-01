import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, learning_curve, StratifiedKFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
iris_feature = iris.data #特征数据
iris_target = iris.target #分类结果

#test_size指定训练集所占的全部数据的百分比，这里采用2/8分法。
#random_state表示乱序程度
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(iris_feature, iris_target):
	feature_train, target_train = iris_feature[train_index], iris_target[train_index]
	feature_test, target_test = iris_feature[test_index], iris_target[test_index]

print(feature_train.shape, target_train.shape)
print(feature_test.shape, target_test.shape)

"""
DecisionTree
"""
dt_model = DecisionTreeClassifier() #默认决策树模型
dt_model.fit(feature_train, target_train) #使用训练集训练模型
dt_results = dt_model.predict(feature_test) #使用模型对测试集进行预测

"""
SVM
"""
svm_pip = Pipeline(
	[
		("scaler", StandardScaler()),
		("svm", svm.SVC(C = 5, gamma = 0.1, kernel = "rbf"))
	]
)

scores = cross_val_score(svm_pip, feature_train, target_train, cv = 3, scoring = "accuracy")
print("scores: {}, scores mean: {} +/- {}".format(scores, np.mean(scores), np.std(scores)))
svm_pip.fit(feature_train, target_train)
svm_results = svm_pip.predict(feature_test)
accuracy = accuracy_score(target_test, svm_results)
print("Accuracy: ", accuracy)
"""
Learing Curve
"""
train_sizes, train_scores, test_scores = learning_curve(
	svm_pip, feature_train, target_train, cv = 3, n_jobs = -1,
	train_sizes = np.linspace(.1, 1.0, 5), scoring = "accuracy"
)
train_scores_mean = np.mean(train_scores, axis = 1)
train_scores_std = np.std(train_scores, axis = 1)
test_scores_mean = np.mean(test_scores, axis = 1)
test_scores_std = np.std(test_scores, axis = 1)

plt.plot(train_sizes, train_scores_mean, "o-", color = "r", label = "Training")
plt.plot(train_sizes, test_scores_mean, "o-", color = "b", label = "Cross-validation")
plt.xlabel('Training')
plt.ylabel('Loss')
plt.title('Learing Curve')
plt.legend(loc = "best")
plt.show()

"""
Validation Curve
"""
X = iris.data
y = iris.target

param_range = np.logspace(-11, -1, 10)

train_loss, test_loss = validation_curve(
	svm.SVC(), X, y, param_name = 'gamma', param_range = param_range, cv = 10, 
	scoring = "accuracy", n_jobs = 1
	)

mean_train_loss = -np.mean(train_loss, axis = 1)
mena_test_loss = -np.mean(test_loss, axis = 1)

plt.semilogx(param_range, mean_train_loss, 'o-', color = 'r', label = 'Training')
plt.semilogx(param_range, mean_train_loss, '--', color = 'b', label = 'Cross-validation')

plt.xlabel('gamma')
plt.ylabel('Loss')
plt.title('Validation Curve')
plt.legend(loc = "best")
plt.show()

"""
ROC Curve
"""
X2 = iris.data
y2 = iris.target
X2, y2 = X2[y2 != 2], y2[y2 != 2]
n_samples, n_features = X.shape
#plot ROC curve and area the curve
cv = StratifiedKFold(n_splits = 6)
classifier = svm.SVC(kernel = 'linear', probability = True, random_state = 42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X2, y2):
	probas_ = classifier.fit(X2[train], y2[train]).predict_proba(X2[test])
	fpr, tpr, thresholds = roc_curve(y2[test], probas_[:, 1])
	tprs.append(np.interp(mean_fpr, fpr, tpr))
	tprs[-1][0] = 0.0
	roc_auc = auc(fpr, tpr)
	aucs.append(roc_auc)
	plt.plot(fpr, tpr, lw = 1, label = 'ROC fold %d (area = %0.2f)' %(i, roc_auc))
	i += 1

#draw diagonal
plt.plot([0, 1], [0, 1], '--', color = 'r',lw = 2, label = 'Luck', alpha = .8)

mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr) #calculate mean auc

#draw mean ROC curve
std_auc = np.std(aucs, axis = 0)
plt.plot(mean_fpr, mean_tpr, '--',color = 'b', label = 'Mean ROC (area = %0.2f)' %mean_auc, lw = 2, alpha = .8)

std_tpr = np.std(tprs, axis = 0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_fpr - std_tpr, 0)
plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color = 'gray', alpha = .2, label = '$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc = "lower right")
plt.show()


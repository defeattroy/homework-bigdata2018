from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier

#下载数据
iris = datasets.load_iris()
x = iris.data
y = iris.target

#将标签二值化
y = label_binarize(y, classes=[0,1,2])

n_classes = y.shape[1]
random_state = np.random.RandomState(0)
n_samples, n_features = x.shape

#分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#训练
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))
y_score = classifier.fit(x_train, y_train).predict_proba(x_test)

#计算所有ROC curve，AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i],_ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#计算总ROC
fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

#画图
lw = 2
plt.figure()
plt.plot(fpr['micro'], tpr['micro'], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc['micro']),
         linestyle=':', linewidth=4)
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(i,roc_auc[i]))
plt.plot([0,1], [0,1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('./curves/ROC_curve.jpg')
plt.show()

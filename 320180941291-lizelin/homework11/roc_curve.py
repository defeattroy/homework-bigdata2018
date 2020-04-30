import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


#引入数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

#将标签二值化
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

#扩大数据规模
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

#划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

#训练模型
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

#计算每一个特征的ROC curve和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#计算微平均的ROC曲线
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#绘图
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('./图片结果/ROC_Curve.jpg')

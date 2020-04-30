from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

"""
下载鸢尾花数据
萼片长度、萼片宽度、花瓣长度、花瓣宽度
"""
iris = load_iris()

"""
data对应了样本的4个特征，150行4列
target对应了样本的类别（目标属性），150行1列
iris.target用0、1和2三个整数分别代表了花的三个品种
"""
X = iris.data
y = iris.target


"""
Draw the ROC curve
"""

"""
将标签二值化
设置种类
"""
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

"""
训练模型并预测
"""
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

"""
分割训练和测试集
Learn to predict each class against the other
通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,random_state=0)
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

"""
计算每一类的ROC
"""
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

"""
fpr, tpr, thresholds  =  roc_curve(y_test, scores) 
y_test为测试集的结果，score为模型预测的测试集得分
fpr,tpr 分别为假正率、真正率
roc_auc =auc(fpr, tpr)
"""
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

"""
Compute macro-average ROC curve and ROC area
First aggregate all false positive rates
"""
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
"""
interpolate all ROC curves at this points
"""
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
"""
average it and compute AUC
"""
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

"""
Plot all ROC curves
"""
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, linewidth=2,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

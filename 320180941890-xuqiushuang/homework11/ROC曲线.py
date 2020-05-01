#ROC曲线
#macro方法：每种类别下，都可以得到m个测试样本为该类别的概率（矩阵P中的列）。所以，根据概率矩阵P和标签矩阵L中对应的每一列，
#可以计算出各个阈值下的假正例率（FPR）和真正例率（TPR），从而绘制出一条ROC曲线。这样总共可以绘制出n条ROC曲线。最后对n条ROC曲线取平均，即可得到最终的ROC曲线。
#原文链接：https://blog.csdn.net/YE1215172385/article/details/79443552

# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import interp

iris = datasets.load_iris()
X = iris.data
y = iris.target
#导入鸢尾花的数据集，并且设定X和y，X指的是各种特征的数据，y指的是分类结果。他们均是np.array形式。
#类别标签排序
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]
#n_classes为有几种分类，这里的n_classes为3
random_state = np.random.RandomState(0)
#设置随机数
n_samples, n_features = X.shape
#对数据进行标准化
sc=StandardScaler()
sc.fit(X)
X_std=sc.transform(X)
#分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.4,stratify=y,random_state=0)
#预测
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

#计算每一类的ROC,fpr:假阳性率,tpr:真阳性率
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#将所有的假阳性率汇总
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#然后插入所有的ROC曲线
#mean_tpr:累计真阳率求平均值
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr,fpr[i],tpr[i])
#最后平均值并计算AUC
#这里通过查询资料发现多分类问题可以通过两种方法绘制ROC曲线，这里我使用了macro方法
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#绘制全部的ROC曲线
lw=4 #线宽
plt.figure(figsize=(20,10))
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='Blue', linestyle='-.', lw=lw)

colors = cycle(['Orange', 'Red', 'Green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0]) # x范围
plt.ylim([0.0, 1.05]) # y范围
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('The ROC curve')
plt.legend(loc="lower right") # label标签放在右下角
plt.show()

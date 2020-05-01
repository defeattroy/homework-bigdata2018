#二分类模型评价 好坏是否概率

import matplotlib.pyplot as plt
from sklearn import datasets,metrics,svm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


#提取数据
iris = datasets.load_iris()
X = iris.data[50:150,:]
y = iris.target[50:150]-1

#7:3分配训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)
clf = svm.SVC(kernel='linear', probability=True, random_state=0)


#用训练集训练模型，预测测试得分
y_score = clf.fit(x_train, y_train).decision_function(x_test)#decisionfunction表示到达超平面的距离
fpr,tpr,threshold = metrics.roc_curve(y_test,y_score)

#输出fpr tpr threshold
data = np.c_[np.c_[fpr,tpr],threshold]
df = pd.DataFrame(data, columns = ['FPR','TPR','threshold'])#threshold阈值
print(df)


auc = metrics.auc(fpr,tpr) #曲线下面积

plt.figure(figsize=(4,4))
plt.plot(fpr,tpr,label='auc={}'.format(auc))
plt.plot([0,1],[0,1],color='g',linestyle=':')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve of iris')
plt.legend(loc='lower right')
plt.show()

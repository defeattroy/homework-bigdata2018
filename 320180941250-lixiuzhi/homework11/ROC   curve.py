
# coding: utf-8

# In[2]:

### ROC curve

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics, model_selection, preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc


# 导入数据，并且变为2分类
iris = datasets.load_iris()
x = iris.data[iris.target !=2, :2]
y_ = iris.target[iris.target !=2]

y=[]
for i in range(len(y_)):
    y.append(y_[i]+1)
y = np.array(y)

# 增加噪声机制
random_state = np.random.RandomState(0)
n_samples, n_features = x.shape
x = np.c_[x, random_state.randn(n_samples, 200 * n_features)]


# 打乱数据集并拆分成训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.3, random_state=25)


# 学习区分个类与其他类
clf = svm.SVC(kernel='linear',probability=True,random_state=random_state)
clf.fit(x_train, y_train)
f1_score = metrics.f1_score(y_test, clf.predict(x_test))
print(f1_score)

predict_probs = clf.predict_proba(x_test)
y_score = predict_probs[:,1]
fpr,tpr,thresholds = metrics.roc_curve(y_test, y_score, pos_label=2)  #计算真正率和假正率
roc_auc = metrics.auc(fpr,tpr)  # 计算auc的值

# 绘制roc曲线
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',lw=lw,label='LR ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




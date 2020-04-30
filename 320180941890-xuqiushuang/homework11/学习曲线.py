#在查询资料的过程中发现，在绘制学习曲线和验证曲线的过程中，我们可以通过管道技术来将数据处理和模型拟合结合在一起，减少代码量。
#Pipeline处理机制就像是把所有模型塞到一个管子里，然后依次对数据进行处理，得到最终的分类结果，pipeline将模型合并成一个模型调用，最后一个模型一定是估计器。
#逻辑回归(Logistic Regression)函数:处理分类问题

#学习曲线
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,learning_curve,validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()
X = iris.data
y = iris.target
#分割训练集与测试集
sc=StandardScaler()
sc.fit(X)
X_std=sc.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.4,stratify=y,random_state=0)
#penalty为正则化选择参数，默认l2正则化
pipe_lr=Pipeline([('scl',StandardScaler()),('clf',LogisticRegression(random_state=1,penalty='l2'))])
#学习曲线
#构建学习曲线评估器，train_sizes：控制用于生成学习曲线的样本的绝对或相对数量,estimator:所使用的分类器
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr,X=X_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)
#统计结果
#train_scores:训练集上的分数，test_scores:在测试集上的分数
train_mean= np.mean(train_scores,axis=1) # 压缩列，对各行求平均值
train_std = np.std(train_scores,axis=1)
test_mean =np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
#绘制效果
plt.figure(figsize=(20,10))
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='test accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7,1.1])
plt.show()

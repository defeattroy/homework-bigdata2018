
# coding: utf-8

# In[5]:

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


#获取数据集
iris=datasets.load_iris()
#将数据分为属性和标签
X=iris.data
y=iris.target
#通过train_test_spli将数据分为训练集、测试集，测试集占0.3的比例
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y,random_state = 1)
#函数make_pipeline是一个构造pipeline的简短工具，他接受可变数量的estimators并返回一个pipeline，每个estimator的名称自动填充
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components = 2), LogisticRegression(random_state = 1))

#定义k折模型
print("Original Class Dist: %s\n" % np.bincount(y))
#分层K折交叉验证
#当我们使用kfold迭代器在k个块中进行循环时，使用train中返回的索引去拟合逻辑斯底回归流水线。通过流水线，我们可以保证
#每次迭代时样本都得到标准化缩放，然后使用test索引计算模型的准确性和f1值，并存放到两个列表中，用于计算平均值和标准差
kfold=StratifiedKFold(n_splits=12,random_state=1).split(X_train,y_train)
 #通过n_splits来设置块的数量
scores=[]
for k,(train_idx,test_idx) in enumerate(kfold):
    pipe_lr.fit(X_train[train_idx],y_train[train_idx])#直接调用fit来对pipeline中的所有算法模型进行训练
    score=pipe_lr.score(X_train[test_idx],y_train[test_idx])
    scores.append(score)
    print("Fold: %2d, Class dist: %s, Acc: %.3f" % (k+1,np.bincount(y_train[train_idx]),score))

print('CV Accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))


scores=cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs = 1)
#计算模型得分
print("\nCV Accuracy Scores: %s" % scores)
print("CV Accuracy: %.3f +/- %.3f" % (np.mean(scores),np.std(np.std(scores))))


# In[ ]:




# In[ ]:




# In[ ]:




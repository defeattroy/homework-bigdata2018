#k-fold找到k为不同值的情况下的分数

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt


#获取数据
iris = load_iris()
X = iris.data
y = iris.target

#数据分类，用训练集训练模型
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)
svm = svm.SVC(kernel='rbf', random_state=0)
std = StandardScaler()#svm模型归一化
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
svm.fit(x_train,y_train)


#设置k的范围
k_range = range(2,16)
mean_score = []


#引入k-fold
for k in k_range:
    #cv = KFold(k,shuffle = True,random_state = 0).get_n_splits(X)
    scores = cross_val_score(svm,x_train,y_train,scoring = 'accuracy',cv=k)#以准确率作为判断标准
    print('{}-fold score mean:'.format(k),scores.mean())
    mean_score.append(scores.mean())


#画出分数折线图
plt.plot(k_range,mean_score)
plt.show()

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve


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
Draw learning curve
"""

features = X[:,0:2] 
labels = y
RF = RandomForestClassifier(max_depth = 8, random_state = 0)
size_grid = np.array([0.2,0.5,0.7,1])
train_size,train_scores,validation_scores = learning_curve(RF,features,labels,train_sizes = size_grid, cv = 5)
"""
学习曲线可视化
"""
plt.figure()
plt.plot(size_grid,np.average(train_scores, axis = 1), color = 'red')
plt.plot(size_grid, np.average(validation_scores, axis = 1), color = 'black')
plt.title('Learning curve')
plt.xlabel('sample size')
plt.ylabel('accuracy')
plt.show()

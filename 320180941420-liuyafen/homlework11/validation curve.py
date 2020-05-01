from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve,learning_curve


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
Draw validation curve
"""

features = X
labels = y
 
RF = RandomForestClassifier(max_depth = 8, random_state = 0)
params_grid = np.linspace(25,200,8).astype(int)
 
"""
其他参数不变，观察评估器数量对训练得分的影响
"""
train_scores,validation_scores = validation_curve(RF,features,labels,'n_estimators',params_grid,cv=5)
"""
可视化生成训练、验证曲线
"""
plt.figure()
plt.plot(params_grid, np.average(train_scores,axis = 1),color = 'red')
plt.plot(params_grid,np.average(validation_scores,axis = 1),color = 'black')
plt.title('Validation curve')
plt.xlabel('number of estimator')
plt.ylabel('accuracy')
plt.show()

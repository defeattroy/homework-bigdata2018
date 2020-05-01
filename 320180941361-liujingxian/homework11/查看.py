from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

iris=datasets.load_iris()

##########

print(iris.data)

print(iris.target)

print(iris.target_names)
'''
########

x=iris.data[:,0]
y=iris.data[:,1]
species=iris.target

x_min,x_max=x.min()-.5,x.max()+.5
y_min,y_max=y.min()-.5,y.max()+.5

plt.figure()
plt.title('Iris DataSet - Classfication By Sepal Sizes')
plt.scatter(x,y,c=species)
plt.xlabel('Speal length')
plt.ylabel('Speal width')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks()
plt.yticks()
plt.show()

########

x=iris.data[:,2]
y=iris.data[:,3]
species=iris.target

x_min,x_max=x.min()-.5,x.max()+.5
y_min,y_max=y.min()-.5,y.max()+.5

plt.figure()
plt.title('Iris DataSet - Classification By Petal Sizes',size=14)
plt.scatter(x,y,c=species)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks()
plt.yticks()
plt.show()

########

x=iris.data[:,1]
y=iris.data[:,2]
species=iris.target
x_reduced=PCA(n_components=3).fit_transform(iris.data)
fig=plt.figure()
ax=Axes3D(fig)
ax.set_title('Iris Dataset By PCA',size=14)
ax.scatter(x_reduced[:,0],x_reduced[:,1],x_reduced[:,2],c=species)
ax.set_xlabel('First eigenvector')
ax.set_ylabel('Second eigenvector')
ax.set_zlabel('Third eigenvector')
ax.w_xaxis.set_ticklabels(())
ax.w_yaxis.set_ticklabels(())
ax.w_zaxis.set_ticklabels(())
plt.show()
'''

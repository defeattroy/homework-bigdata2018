import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_data=pd.read_csv('抑郁症数据.csv',index_col=[0])
#计算时间删除异常值
time=df_data.loc[df_data['type']=='accelerometer']
time=time.groupby('number').count()#计算
time=time[['x']]
#计算时间,频率为5hz
time['x']=time['x']/300 #单位为min
#画箱式图查找异常值
time1= time.boxplot(return_type='dict') #画箱线图，直接使用DataFrame的方法
x = time1['fliers'][0].get_xdata() # 'flies'即为异常值的标签
y = time1['fliers'][0].get_ydata()
for i in range(len(x)):
  if i>0:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
  else:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))
plt.show() #展示箱线图
#根据箱式图获取异常值
y.sort()
result = time[time['x']>=y[0]].index.tolist()
#剔除异常值
df_data=df_data[~df_data['number'].isin(['result'])]

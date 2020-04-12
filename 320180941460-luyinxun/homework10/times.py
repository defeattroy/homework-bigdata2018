import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from numpy as np

'''
@data1: the data of health people
@data2: the data of anxiety people

three types of data: gyroscope陀螺仪、accelerometer加速度、device motion设备运动
two types of people: health and anxiety
'''


d = int(input('获取数据\n（0：陀螺仪  1：加速度  2：位置数据）\n'))
l = ['gyroscope','accelerometer','device']


filenames = listdir('D:\数据集\数据')
times_health = []
times_anxiety = []
for i in range(len(filenames)):
    time = 0
    df = pd.read_json('d:\\数据集\\数据\\' + filenames[i])  #get the file
    
    time = df.shape[0] / (5*60)      #calculate the time
    
    if filenames[i].split('_')[0].startswith(l[d]):  #choose the right files
        if 'health' in filenames[i]:
            times_health.append(time)
        if 'anxiety' in filenames[i]:
            times_anxiety.append(time)
            
data1 = pd.DataFrame(times_health, columns = ['health'])  
data2 = pd.DataFrame(times_anxiety,columns = ['anxiety'])
data1['health']=pd.to_numeric(data1['health'])     #convert the type of data into float
data2['anxiety']=pd.to_numeric(data2['anxiety'])


#图形化输出查看异常   
print(data1.hist())
print(data2.hist())
print(data1.boxplot())
print(data2.boxplot())


#处理零值，转化为nan再删除
data1[data1 == 0.0] = np.nan
data1 = data1.dropna()

data2[data2 == 0.0] = np.nan
data2 = data2.dropna()
print(data1,data2)


#根据箱形图数据提取异常值
high = data1.health.quantile(q=0.75)
low = data1.health.quantile(q=0.25)
data1 = data1.loc[data1.health > high + 1.5*(high - low)] 
print(data1)

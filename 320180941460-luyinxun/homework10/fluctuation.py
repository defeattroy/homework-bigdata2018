import pandas as pd
from os import listdir


#删除空文件
dirpath = 'd:\\数据集\\数据'
file = listdir(dirpath)
for i in file:
    df = pd.read_json(dirpath + '\\' + i)
    if df.empty:
        print(i)
        os.remove(dirpath + '\\' + i)


#提取数据并画出所有折线图
filenames = input('请输入关键字（gyr acc dev）：')

if filenammes.statswith('dev'):
    j = 2
else:
    j =1
    
f = listdir(dirpath)
health = []       
anxiety = []

for i in range(len(f)):
    if f[i].startswith(filenames):
        
        '''distinguish the data of health and anxiety'''
        
        if f[i].split('_')[j].startswith('hea'):
            health.append(f[i])
        else:
            anxiety.append(f[i])
                
print((len(dev_health),len(dev_anxiety)))
for i in anxiety:
    df = pd.read_json(dirpath + '\\' + i)
    print(df.plot())

#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import os
#获得一个可进行数据分析的文件
def GetFile():
    flag=1
    while (flag):
        filename = input("请输入要处理的文件名:")
        if filename not in os.listdir('.'):
            print("文件不存在请重新输入")
            flag=1
        elif os.path.getsize(filename) < 2048:
            print("文件过小请重新输入")
            flag=1
        else:
            print(filename+'文件可进行数据分析')
            flag=0
    return filename
filename=GetFile()


# In[10]:


#数据变换
data = pd.read_json(filename)


# In[12]:


#数据预览
data


# In[14]:


#数据可视化
data.plot()


# In[29]:


#空值用均值填充
if(data.isnull().sum().sum()!=0):
    _x = data.mean(axis=0)[0]           
    _y = data.mean(axis=0)[1]
    _z = data.mean(axis=0)[2]
    data.fillna({'x':_x,'y':_y,'z':_z})


# In[31]:


#筛除时间过长或者过短数据
time = data.shape[0]/5/60
if time > 90 or time < 15:
    print("答题时长少于15min或大于90min,数据无意义")


# In[33]:


#筛除没用的数据
if data[data.columns[0]].var() < 0.001 or data[data.columns[1]].var() < 0.001 or data[data.columns[2]].var() < 0.001:
    print("大概率没有动，数据无意义")


# In[ ]:


#得到有意义数据
print('数据有意义')


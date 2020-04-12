#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## 下载文件

# In[2]:


response = requests.get('http://yang.lzu.edu.cn/data/index.txt')
data_name = response.text.split()
# print(data_name)
data_url_list = ['http://yang.lzu.edu.cn/data/{}'.format(i.lstrip(r'./')) for i in data_name if len(i) > len('./device_motion/anxiety/female')]  # 拼接URL完成
# print(data_url_list)  # 所有文件的url


# In[3]:


def graph(x):
    plt.figure()
    plt.plot(x)
    plt.show()


# In[5]:


def var_len(df: pd.DataFrame)->tuple:
    x_var, y_var, z_var = np.std(df.iloc[:, 0]), np.var(df.iloc[:, 1]), np.var(df.iloc[:, 2])
    return x_var, y_var, z_var, len(df)


# In[6]:


def inner_preprocessing(df):
    # 空值，缺失值-->加速度用相邻的前一个值来填充
    return df.replace([None], np.NaN).fillna(method='ffill').fillna(method='bfill')  


# In[7]:


def to_dataframe(raw_dt:list):  # 把一个文件转换为一个dataframe，col为
    data_col = list(raw_dt[0].keys())
    data_values = [list(i.values()) for i in raw_dt]
    return pd.DataFrame(data=data_values, columns=data_col)


# In[8]:


def write_file(file_name, file_content):
    file_path = os.path.split(file_name)[0]
    if not os.path.exists(file_path):  # 生成文件目录
        os.makedirs(file_path)
    with open(file_name, 'wb') as f:  # 在对应目录下写入文件
        pickle.dump(file_content, f)


# In[9]:


# 下载文件+处理空值
file_num = 0
empty_list = []
for num, data_url in enumerate(data_url_list):
    print(str(num+1)+'. from', data_url)
    file_content = requests.get(data_url)
#     print(file_content.text)
    if file_content.text == r'[]': # 有的文件内容为空，把这些文件在下载的同时就去除掉，减少之后的工作量
        empty_list.append(data_url)
    else:
        # 在写入文件之前，完成内部空值的替换，计算时间长度和方差。
        raw_df = to_dataframe(eval(file_content.text))
        raw_df = inner_preprocessing(raw_df)  # 对内部的空值，缺失值进行替换
#         print(raw_df.values)  
            # 得到x,y,z方差，df长度放到文件末尾
#         print(raw_df.values)
        df_ready = raw_df.values.tolist()
        
        df_ready.append(var_len(raw_df))
#         print(df_ready)
        
        
        # 用pickle写入文件，这样数据读出来就可以直接用
        write_file('./'+data_url[23:].rstrip('.json'), df_ready)
        print('write successful: ./'+data_url[23:].rstrip('.json'))
    file_num += 1
    print()
#     break


# In[10]:


print('download {} files of {}'.format(file_num - len(empty_list), len(data_url_list)), end=', ')
print('{} empty files are ignored:\n'.format(len(empty_list)))
for i in empty_list:  # 打出没有内容的文件名
    print(i)


# ## 数据预处理

# 需要预处理的地方：
# 
# 1、空值和缺失值（文件内部）-->下载时已经处理了
# 
# 2、过短、长的时间（文件级别）--> 短的删掉，长的截断
# 
# 3、变化范围较小，放在桌子上的（文件级别）--> 删除
# 
# 4、开始站着，后来坐着（文件级别）-->据观察，截掉前250个数据可以把大多数这种情况排除掉

# In[11]:


def show_file(df, bool_index):
    '''
    输入dataframe和某一行的布尔索引，会画出其中某个文件的图
    '''
    if bool_index is not None and any(bool_index.tolist()) is True:
        fn = df[bool_index]['fname']
        with open(fn.tolist()[0], 'rb') as f:
            content = pickle.load(f)[:-1]
        plt.figure()
        plt.title(fn.tolist()[0])
        plt.plot(content)
        plt.legend(['x', 'y', 'z'])
        plt.show()

        
        
def show_dataframe(df, del_list=None):
    '''
    输入一个dataframe，会把里面的每一个文件都画出来
    del_list是标记被舍弃的文件的时间戳，删除前可以让人确认一下
    '''
    print('共'+str(len(df))+'个')
    if del_list is None:
        del_list = []
    for i in range(len(df)):
        print(i+1)
        if find_timestamp(df['fname'][i]) in del_list:
            print('-----droped-----')
        show_file(df, df['y_var'] == df['y_var'][i])

def find_full_name(df, text):
    '''
    根据部分文件名，返回文件的全名
    '''
    return df[df.fname.str.contains(text)]['fname'].tolist()[0]

def find_timestamp(text, num=5, lt=19):
    '''
    >>> text = './data/device_motion/heath/female/20191110161313_1383_dawbdauda'
    >>> find_timestamp(text) # 找到目录中的时间戳
    '20191110161313_1383'
    '''
    cur_n = 0
    for i in range(num):
        cur_n = text.find(r'/', cur_n+1)
    return text[cur_n+1:cur_n+1+lt]

def add_d_list(df, num_list, d_list):
    for i in range(len(df)):
        if i+1 in num_list:  # 如果是要被删除的文件
            d_list.append(find_timestamp(df['fname'].tolist()[i]))


# ###  把文件内容读到total_dict中

# In[13]:


paths = []
root_dir = './data/'
tar_dirs = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if not i.startswith('.')]
# print(tar_dirs)  # ['./data/device_motion', './data/accelerometer', './data/gyroscope']
for tar_dir in tar_dirs:
    paths.append([os.path.join(tar_dir, i, 'female') for i in os.listdir(tar_dir) if not i.startswith('.')])
# print(paths)
'''
[['./data/device_motion/anxiety/female', './data/device_motion/health/female'], ['./data/accelerometer/anxiety/female', './data/accelerometer/health/female'], ['./data/gyroscope/anxiety/female', './data/gyroscope/health/female']]
'''
# paths[0]--device motion
# paths[1]--accelerometer
# paths[2]--gyroscope
total_dict = {'device_motion':None, 'accelerometer':None, 'gyroscope':None}

for cat_num, cat_name in enumerate(total_dict.keys()):
#     print(cat_num,cat_name)
    cat_dict = {'anxiety':{'fname':[], 'detail':[]}, 'health':{'fname':[], 'detail':[]}}
    
    cat_dict['anxiety']['fname'] =[os.path.join(paths[cat_num][0], i) for i in os.listdir(paths[cat_num][0]) if not i.startswith('.')]
    cat_dict['health']['fname'] = [os.path.join(paths[cat_num][1], i) for i in os.listdir(paths[cat_num][1]) if not i.startswith('.')]
#     print(cat_dict)
    
    for cat in cat_dict.keys():
    #     print(cat)  # anxiety, health
        for f_num in range(len(cat_dict[cat]['fname'])):
                with open(cat_dict[cat]['fname'][f_num], 'rb') as f:
                    cat_dict[cat]['detail'].append(pickle.load(f)[-1])
    total_dict[cat_name] = cat_dict.copy()


# In[14]:


total_dict  # 到此为止，所有文件的信息都在这个字典中了


# ### 探索3种传感器的数据

# In[57]:


ac_ax = set([i.lstrip('./data/accelerometer/anxiety/female/').rstrip('_accelerometer') for i in total_dict['accelerometer']['anxiety']['fname']])
ac_ht = set([i.lstrip('./data/accelerometer/health/female/').rstrip('_accelerometer') for i in total_dict['accelerometer']['health']['fname']])
dm_ax = set([i.lstrip('./data/device_motion/anxiety/female/').rstrip('_device_moti') for i in total_dict['device_motion']['anxiety']['fname']])
dm_ht = set([i.lstrip('./data/device_motion/health/female/').rstrip('_device_moti') for i in total_dict['device_motion']['health']['fname']])
gr_ax = set([i.lstrip('./data/gyroscope/anxiety/female/').rstrip('_gyroscope') for i in total_dict['gyroscope']['anxiety']['fname']])
gr_ht = set([i.lstrip('./data/gyroscope/health/female/').rstrip('_gyroscope') for i in total_dict['gyroscope']['health']['fname']])

# 看一下抑郁症和健康的参与者总共有多少人

len(gr_ax | ac_ax | dm_ax)  # 44个  
len(gr_ht | ac_ht | dm_ht)  # 47个

len(ac_ht)  # accelerometer中health的参与者人数：41
len(ac_ax)  # accelerometer中anxiety的参与者人数：37

len(dm_ht)  # device_motion中health的参与者人数：33
len(dm_ax)  # device_motion中anxiety的参与者人数：34

len(gr_ax)  # gyroscope中anxiety的参与者人数：21
len(gr_ht)  # gyroscope中health的参与者人数：28

# 可见，有accelerometer传感器的人占大多数


# In[23]:


d_list = []  # 把发现的异常文件加入这个列表之中


# #### accelerometer抑郁

# In[24]:


df_ac_anxiety = pd.DataFrame(data=total_dict['accelerometer']['anxiety']['detail'], columns=['x_var', 'y_var', 'z_var', 'len'])
df_ac_anxiety['fname'] = total_dict['accelerometer']['anxiety']['fname']
df_ac_anxiety.boxplot(column=['x_var', 'y_var','z_var'])


# ##### 查看答题时间分布

# In[25]:


plt.figure()
plt.hist(df_ac_anxiety['len']/(5*60))
plt.title('Anxiety answer time')
plt.show()


# ##### 检查运动幅度

# In[26]:


df_ac_anxiety = df_ac_anxiety.sort_values(by='x_var').reset_index()  
# 按照方差排序，把方差小的放到前面，通过画图，人工检测是不是把手机放到桌子上了
df_ac_anxiety


# In[27]:


show_dataframe(df_ac_anxiety)


# ##### 删除前确认

# 可见第2、3号文件变化比较小，需要被删除的

# In[28]:


add_d_list(df_ac_anxiety, [2, 3], d_list)
d_list


# In[29]:


show_dataframe(df_ac_anxiety, d_list)


# #### accelerometer健康

# In[30]:


df_ac_health = pd.DataFrame(data=total_dict['accelerometer']['health']['detail'], columns=['x_var', 'y_var', 'z_var', 'len'])
df_ac_health['fname'] = total_dict['accelerometer']['health']['fname']
df_ac_health.boxplot(column=['x_var', 'y_var','z_var'])  # xyz轴方差


# ##### 查看答题时间分布

# In[31]:


plt.figure()
plt.hist(df_ac_health['len']/(5*60))  # 换算成分钟
plt.title('Health answer time')
plt.show()


# ##### 检查运动幅度

# In[32]:


df_ac_health = df_ac_health.sort_values(by='x_var').reset_index()  
# 按照方差排序，把方差小的放到前面，通过画图，人工检测是不是把手机放到桌子上了(x的变化范围大)
df_ac_health


# In[33]:


show_dataframe(df_ac_health)


# 1、30号幅度过小，需要被处理掉

# ##### 删除前确认

# In[34]:


add_d_list(df_ac_health, [1, 30], d_list)
d_list


# In[35]:


show_dataframe(df_ac_health, d_list)


# #### device motion抑郁

# In[36]:


df_dm_anxiety = pd.DataFrame(data=total_dict['device_motion']['anxiety']['detail'], columns=['x_var', 'y_var', 'z_var', 'len'])
df_dm_anxiety['fname'] = total_dict['device_motion']['anxiety']['fname']
df_dm_anxiety.boxplot(column=['x_var', 'y_var','z_var'])


# ##### 检查运动幅度

# In[37]:


df_dm_anxiety = df_dm_anxiety.sort_values(by='y_var').reset_index()
df_dm_anxiety


# In[38]:


show_dataframe(df_dm_anxiety, d_list)


# 第1、2、9、25个需要被删除

# ##### 删除前确认

# In[39]:


add_d_list(df_dm_anxiety, [1, 2, 9, 25], d_list)
d_list
show_dataframe(df_dm_anxiety, d_list)


# In[ ]:





# In[ ]:





# #### device motion健康

# In[40]:


df_dm_health = pd.DataFrame(data=total_dict['device_motion']['health']['detail'], columns=['x_var', 'y_var', 'z_var', 'len'])
df_dm_health['fname'] = total_dict['device_motion']['health']['fname']
df_dm_health.boxplot(column=['x_var', 'y_var','z_var'])


# ##### 检查运动幅度

# In[41]:


df_dm_health = df_dm_health.sort_values(by='y_var').reset_index()
df_dm_health


# In[42]:


show_dataframe(df_dm_health, d_list)


# 第3，6，20，30个需要被删除

# ##### 删除前确认

# In[43]:


add_d_list(df_dm_health, [3, 6, 20, 30], d_list)
show_dataframe(df_dm_health, d_list)


# In[44]:


d_list


# #### gyroscope抑郁

# In[45]:


df_gr_anxiety = pd.DataFrame(data=total_dict['gyroscope']['anxiety']['detail'], columns=['x_var', 'y_var', 'z_var', 'len'])
df_gr_anxiety['fname'] = total_dict['gyroscope']['anxiety']['fname']
df_gr_anxiety.boxplot(column=['x_var', 'y_var','z_var'])


# ##### 检查运动幅度

# In[46]:


df_gr_anxiety = df_gr_anxiety.sort_values(by='x_var').reset_index()
df_gr_anxiety


# In[47]:


show_dataframe(df_gr_anxiety, d_list)


# 陀螺仪数据未发现明显异常

# #### gyroscope健康

# In[48]:


df_gr_health = pd.DataFrame(data=total_dict['gyroscope']['health']['detail'], columns=['x_var', 'y_var', 'z_var', 'len'])
df_gr_health['fname'] = total_dict['gyroscope']['health']['fname']
df_gr_health.boxplot(column=['x_var', 'y_var','z_var'])


# ##### 检查运动幅度

# In[49]:


df_gr_health = df_gr_health.sort_values(by='x_var').reset_index()
df_gr_health


# In[50]:


show_dataframe(df_gr_health, d_list)


# 陀螺仪数据未见明显异常，除了根据其他传感器数据判断的需要删除的文件外不需要删除

# ### 处理异常文件

# 思路：
# 
# 1、截断:  (1)如果时间小于8分钟，或者长于60分钟，则把在范围之外的数据截掉
#         
#         (2)删除每个文件前300个记录，把开始的波动去掉。
# 
# 2、删除：变化幅度过小的文件
# 
# 方法：
# 
# 需要删除的文件已经加入了d_list，然后重新下载一遍，在下载的过程中排除上述列表中的文件（不下载=删除），即可得到处理完成的数据。

# In[51]:


d_list  # 要删除的数据如下


# In[52]:


# os.system('mv data data_origin')  # 把原来的文件改名，准备下载新的文件（linux环境下的命令）


# In[53]:


# 下载文件+处理数据
n_min = 60
file_num = 0
empty_list = []
del_list = []
for num, data_url in enumerate(data_url_list):
    print(str(num+1)+'. from', data_url)
    file_content = requests.get(data_url)
#     print(file_content.text)
    if file_content.text == r'[]': # 有的文件内容为空，把这些文件在下载的同时就去除掉，减少之后的工作量
        empty_list.append(data_url)
        print('empty')
    elif find_timestamp(data_url, 7) in d_list:
        del_list.append(data_url)
        print('deleted')
    else:
        # 在写入文件之前，完成内部空值的替换，计算时间长度和方差。
        raw_df = to_dataframe(eval(file_content.text))
        raw_df = inner_preprocessing(raw_df)  # 对内部的空值，缺失值进行替换
            
#         print(raw_df.values)
        df_ready = raw_df.values.tolist()[300:] if len(raw_df.values.tolist()) <= 5*60*n_min else raw_df.values.tolist()[250:5*60*n_min]
#         df_ready.append(var_len(raw_df))  # 得到x,y,z方差，df长度放到文件末尾
#         print(df_ready)
        
        # 用pickle写入文件，这样数据读出来就可以直接用
        write_file('./'+data_url[23:].rstrip('.json'), df_ready)
        
        a = []
        with open(data_url[23:].rstrip('.json'), 'rb') as f:
            a = pickle.load(f)
        data1 = pd.DataFrame(a, columns=['x', 'y', 'z'])
        graph(data1)
        
        print('write successful: ./'+data_url[23:].rstrip('.json'))
    file_num += 1
    print()


# In[54]:


print('download {} files of {}'.format(file_num - len(empty_list) - len(del_list), len(data_url_list)), end=', ')
print('{} empty files are ignored:\n'.format(len(empty_list)))
for i in empty_list:  # 打出没有内容的文件名
    print(i)
print('{} abnormal files are deleted:\n'.format(len(del_list)))
for i in del_list:
    print(i)


# In[ ]:





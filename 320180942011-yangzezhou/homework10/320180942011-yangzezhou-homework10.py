# -*- coding: utf-8 -*-
# @Time    : 2020.4.11
# @Author  : Yang Zezhou
# @FileName: homework10.py
# @Software: PyCharm

import urllib.request
import os
import pandas as pd
import matplotlib.pyplot as plt



def ret_previous_dir(n):
    '''
    You can return the previous dictionary with a certain number.
    '''
    for i in range(n):
        os.chdir('..')

def getdir():
    '''
    You can get all the useful json files by the function.
    '''
    dirs = os.listdir()
    for dirr in dirs:
        if not dirr.endswith('.json'):
            dirs.remove(dirr)
        else:
            if dirr.endswith('error.json'):
                dirs.remove(dirr)
    return dirs

def getDataFrame(dirs):
    '''
    This is a function to make all the json files transform DataFrame in a certain dictionary.
    '''
    df = []
    for dirr in dirs:
        try:
            df.append(pd.read_json(dirr))
        except Exception as e:
            print(e)
    return df

def error_file(indexes,dirs):
    '''
    In the dictionary, mark 'error' for the useless files.
    '''
    try:
        for i in indexes:
            os.rename(dirs[i - 1], dirs[i - 1][:-5] + '_error.json')
    except Exception as e:
        print(e)

def time_error_file(indexes,dirs):
    try:
        for i in indexes:
            os.rename(dirs[i - 1], dirs[i - 1][:-5] + '_time_error.json')
    except Exception as e:
        print(e)

def download_doc():
    '''
    It is a crawl to get the data.Use the function, we can download the data to our computer.
    '''
    path = 'http://yang.lzu.edu.cn/data'
    try:
        req = urllib.request.Request(path+'/index.txt')#可以将url先构造成一个Request对象，传进urlopen
        f = urllib.request.urlopen(req)
    except Exception as e:
        print(e)
    count = 0
    content = ''
    file = ''
    line = str(f.readline(),encoding='utf8').strip('\n')
    while line:
        line = line.replace('/','\\')
        if line.endswith('.json'):
            file = line[line.rfind('\\')+1:]
            tempath = (path + content[1:] + '/' + file).replace('\\','/')
            system_path = '.\\' + file
            try:
                req_t = urllib.request.Request(tempath)
                f_t = urllib.request.urlopen(req_t)
                f_str = f_t.read()
                if not os.path.exists(system_path):
                    with open(system_path,'wb') as paper:
                        paper.write(f_str)
                        paper.close()
            except Exception as e:
                print(e)
        else:
            if count>0:
                ret_previous_dir(3)
            content = line
            if not os.path.exists(content):
                os.makedirs(content)
            count += 1
            print('count',count)
            os.chdir(content)
            print(os.getcwd())
            print('content:',content)
        line = str(f.readline(),encoding='utf8').strip('\n')
    ret_previous_dir(3)
    print('All the files have loaded in your compter on ',os.getcwd())

download_doc()

os.chdir('.\\accelerometer\\anxiety\\female')

dirs = getdir()
df = getDataFrame(dirs)
for dff in df:
    # Make the DataFrames have the same index like x,y,z.
    if not all(dff.columns.values == ['x','y','z']):
        dff = dff.reindex(columns=['x','y','z'])
    # There will occur all the pictures, which we can find what pictures are not suitable.
    #dff.plot()
    #plt.show()
# From the front picture, we can get the useless pictures' index(By my eye). And we will delete them.
error_file([11,24],dirs)
dirs = getdir()
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
# From the bargram and boxgram, we can find the useless time and we think it is the useless data.
plt.bar(longx,longy)
plt.show()
plt.boxplot(longy)
plt.show()
time_error_file([1,7,11,14,23,34],dirs)
dirs = getdir()
del dirs[8]
del dirs[18]
df = getDataFrame(dirs)
print(len(df))
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
plt.boxplot(longy)# You can see all the data have been formal.
plt.show()
plt.bar(longx,longy)
plt.show()



ret_previous_dir(2)
os.chdir('.\\health\\female')

dirs = getdir()
df = getDataFrame(dirs)
for dff in df:
    # Make the DataFrames have the same index like x,y,z.
    if not all(dff.columns.values == ['x','y','z']):
        dff = dff.reindex(columns=['x','y','z'])
    # There will occur all the pictures, which we can find what pictures are not suitable.
    #dff.plot()
    #plt.show()
# From the front picture, we can get the useless pictures' index(By my eye). And we will delete them.
# The index is (16,19,27,32)
error_file([16,19,27,32],dirs)
dirs = getdir()
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
# From the bargram and boxgram, we can find the useless time and we think it is the useless data.
plt.bar(longx,longy)
plt.show()
plt.boxplot(longy)
plt.show()
time_error_file([6,9,17,25,26,29,32],dirs)
dirs = getdir()
del dirs[14]
del dirs[21]
del dirs[23]
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
plt.boxplot(longy)# You can see all the data have been formal.
plt.show()
plt.bar(longx,longy)
plt.show()



ret_previous_dir(3)
os.chdir('.\\device_motion\\anxiety\\female')

dirs = getdir()
df = getDataFrame(dirs)
for dff in df:
    # Make the DataFrames have the same index like alpha,beta,gamma.
    if not all(dff.columns.values == ['alpha', 'beta', 'gamma']):
        dff = dff.reindex(columns=['alpha', 'beta', 'gamma'])
    # There will occur all the pictures, which we can find what pictures are not suitable.
    #dff.plot()
    #plt.show()
# From the front picture, we can get the useless pictures' index(By my eye). And we will delete them.
# The index is (5,10,11,12,13,22,23)
error_file([5,10,11,12,13,22,23],dirs)
dirs = getdir()
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
# From the bargram and boxgram, we can find the useless time and we think it is the useless data.
# You can see all the data have been formal.
# It is formal, so you don't need to deal with them.
plt.boxplot(longy)
plt.show()
plt.bar(longx,longy)
plt.show()



ret_previous_dir(2)
os.chdir('.\\health\\female')

dirs = getdir()
df = getDataFrame(dirs)
for dff in df:
    # Make the DataFrames have the same index like alpha,beta,gamma.
    if not all(dff.columns.values == ['alpha', 'beta', 'gamma']):
        dff = dff.reindex(columns=['alpha', 'beta', 'gamma'])
    # There will occur all the pictures, which we can find what pictures are not suitable.
    #dff.plot()
    #plt.show()
# From the front picture, we can get the useless pictures' index(By my eye). And we will delete them.
# The index is (13,18,20,30)
error_file([13,18,20,30],dirs)
dirs = getdir()
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
# From the bargram and boxgram, we can find the useless time and we think it is the useless data.
plt.bar(longx,longy)
plt.show()
plt.boxplot(longy)# You can see all the data have been formal.
plt.show()



ret_previous_dir(3)
os.chdir('.\\gyroscope\\anxiety\\female')

dirs = getdir()
df = getDataFrame(dirs)
for i in range(len(df)):
    # There are some empty files. We will delete them directly.
    if df[i].empty:
        os.remove(dirs[i])
    # Make the DataFrames have the same index like x,y,z.
    else:
        if not all(df[i].columns.values == ['x','y','z']):
            df[i] = df[i].reindex(columns=['x','y','z'])
        # There will occur all the pictures, which we can find what pictures are not suitable.
        #dff.plot()
        #plt.show()
# From the front picture, we can get the useless pictures' index(By my eye). And we will delete them.
# The index is the number 1.
error_file([1],dirs)
dirs = getdir()
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
# From the bargram and boxgram, we can find the useless time and we think it is the useless data.
plt.bar(longx,longy)
plt.show()
plt.boxplot(longy)
plt.show()



ret_previous_dir(2)
os.chdir('.\\health\\female')

dirs = getdir()
df = getDataFrame(dirs)
for i in range(len(df)):
    # There are some empty files. We will delete them directly.
    if df[i].empty:
        os.remove(dirs[i])
    # Make the DataFrames have the same index like x,y,z.
    else:
        if not all(df[i].columns.values == ['x','y','z']):
            df[i] = df[i].reindex(columns=['x','y','z'])
        # There will occur all the pictures, which we can find what pictures are not suitable.
        #dff.plot()
        #plt.show()
# From the front picture, we can get the useless pictures' index(By my eye). And we will delete them.
# The index is the number 8.
error_file([8],dirs)
dirs = getdir()
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
# From the bargram and boxgram, we can find the useless time and we think it is the useless data.
plt.bar(longx,longy)
plt.show()
plt.boxplot(longy)
plt.show()
time_error_file([5,12,14,15,21,27],dirs)
dirs = getdir()
del dirs[11]
print(len(dirs))
df = getDataFrame(dirs)
longy = []
for i in range(len(df)):
    longy.append(len(df[i]))
longx = list(range(len(longy)))
plt.boxplot(longy)# You can see all the data have been formal.
plt.show()
plt.bar(longx,longy)
plt.show()

ret_previous_dir(3)
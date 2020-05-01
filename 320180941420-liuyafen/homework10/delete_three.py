import json
import pandas as pd
import numpy as np
import xlrd
import os

"""
将json文件转换成excel表
excel表的文件名跟它的json的文件名相同，只是后缀为.xlsx
这一步是为了算方差
"""

def json_excel(s):
    data = [] # 用于存储每一行的Json数据
    with open(s,'r', encoding = 'UTF-8') as fr:
        for line in fr:
            j = json.loads(line)
            data.append(j)

    df = pd.DataFrame() # 最后转换得到的结果
    for line in data:
        for i in line:
            df1 = pd.DataFrame([i])
            df = df.append(df1)

    b = s[:-5] + '.xlsx'
    df.to_excel(b, startcol=0, index=False)
    
    fr.close()


"""
根据表中的数据计算方差
"""
def Calculated_variance(s):
    mean = [] #存储方差
    data = xlrd.open_workbook(s)
    table = data.sheet_by_name(u'Sheet1')#通过名称获取
    nrows = table.nrows#获取行号
    ncols = table.ncols#获取列号
    
    col_one = []  # 建立一个数组来存储数据
    col_two=[]
    col_three=[]

    for i in range(1, nrows):#第0行为表头
        alldata = table.row_values(i)#循环输出excel表中每一行，即所有数据
        col_one.append(float(alldata[0]))  # 将字符串数据转化为浮点型加入到数组之中
        col_two.append(float(alldata[1]))
        col_three.append(float(alldata[2]))
    
    mean.append(abs(np.mean(col_one))) # 输出方差的绝对值
    mean.append(abs(np.mean(col_two)))
    mean.append(abs(np.mean(col_three)))
    
    print(mean)  #打印出方差
    
    return mean


"""
删除方差过小的文件
a 代表x方差的大小
b 代表y方差的大小
c 代表z方差的大小
x 代表衡量x的方差，即与a做比较，如果a<x就将这个文件删除
y 代表衡量y的方差，即与b做比较，如果b<y就将这个文件删除
z 代表衡量z的方差，即与c做比较，如果c<z就将这个文件删除
s 代表文件路径
"""

def delete_data(a, b, c, x, y, z, s):  # s是文件名(字符串)
    if a<x or b<y or c<z:
        os.remove(s)
        print(s  + " deleted.")


"""
该函数是为了删除方差太小的文件
调用json_excel(s)函数，将json文件写入表中，
再调用Calculated_variance(s)函数计算方差，
用delete_data(a, b, c, x, y, z, s)函数将满足条件的json文件删除
"""

def deleteFile_three(x, y, z, s):
    files = os.listdir(s)
    for file in files:
        if 'json' in file:
            json_excel(s + "//" + file)
            i = []
            i = Calculated_variance(s + "//" + file[:-5] + '.xlsx')
            os.remove(s + "//" + file[:-5] + '.xlsx')  #删除用掉的excel表
            delete_data(i[0],i[1],i[2],x,y,z,s + "//" + file)


way1 = 'C:\\homework10\\accelerometer\\anxiety'
x1 = 0.01
y1 = 0.1
z1 = 0.5
way2 = 'C:\\homework10\\accelerometer\\health'
x2 = 0.02
y2 = 0.1
z2 = 0.2
way3 = 'C:\\homework10\\device_motion\\anxiety'
x3 = 40.0
y3 = 10.0
z3 = 1.0
way4 = 'C:\\homework10\\device_motion\\health'
x4 = 40.0
y4 = 10.0
z4 = 1.0
way5 = 'C:\\homework10\\gyroscope\\anxiety'
x5 = 0.0001
y5 = 0.0001
z5 = 0.0001
way6 = 'C:\\homework10\\\gyroscope\\health'
x6 = 0.0001
y6 = 0.0001
z6 = 0.0001

if __name__=='__main__':
    doctest.testmod(verbose=True)
    deleteFile_three(x1, y1, z1, way1)
    deleteFile_three(x2, y2, z2, way2)
    deleteFile_three(x3, y3, z3, way3)
    deleteFile_three(x4, y4, z4, way4)
    deleteFile_three(x5, y5, z5, way5)
    deleteFile_three(x6, y6, z6, way6)

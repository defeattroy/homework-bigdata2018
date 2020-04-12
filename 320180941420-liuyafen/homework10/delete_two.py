import os
import pandas as pd
import json

"""
计算每个文件里问卷填写者所用的的时间
用save_dict字典存储文件名和对应的时间
"""

def Calculated_time(s):
    save_dict = {}
    files = os.listdir(s)
    for file in files:
        b = s + "//" + file
        save = pd.read_json(b)
        save_dict[file] = save.shape[0]
    
    for key,value in save_dict.items():
        save_dict[key] = value/5
    
    return save_dict


"""
根据计算出来的时间删除小于10分钟大于110分钟的数据
函数的参数s是字符串型，代表文件路径
"""

def deleteFile_two(s):
    s_d = Calculated_time(s)
    for key,value in s_d.items():
        if s_d[key] > 110*60 or s_d[key] < 10*60:
            b = s + '//' + key
            os.remove(b)
            print(key  + " deleted.")


"""
调用函数删除6个文件夹里的不符合时间标准的文件
"""

way1 = 'C:\\homework10\\accelerometer\\anxiety'
way2 = 'C:\\homework10\\accelerometer\\health'
way3 = 'C:\\homework10\\device_motion\\anxiety'
way4 = 'C:\\homework10\\device_motion\\health'
way5 = 'C:\\homework10\\gyroscope\\anxiety'
way6 = 'C:\\homework10\\\gyroscope\\health'

if __name__=='__main__':
    doctest.testmod(verbose=True)
    deleteFile_two(way1)
    deleteFile_two(way2)
    deleteFile_two(way3)
    deleteFile_two(way4)
    deleteFile_two(way5)
    deleteFile_two(way6)

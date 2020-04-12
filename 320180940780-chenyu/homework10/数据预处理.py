import os
import pandas as pd


#删除内部为空的文件
def del_size(file_path):
    for root, dirs, files in os.walk(file_path):
        for file in files:
            filename = os.path.join(root, file)
            size = os.path.getsize(filename)
            if size <= 1024:
                os.remove(filename)
                print("remove", filename)

#删除测试时间过长或过短的文件
def del_time(file_path):
    for root, dirs, files in os.walk(file_path):
        for file in files:
            filename = os.path.join(root, file)
            fp1 = pd.read_json(filename)
            time = fp1.shape[0]/300
            if(time<10.0 or time>60.0):
                os.remove(filename)
                print('remove',filename)
                
#以z轴为标准删除方差较小的文件
def del_agz(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            fp1 = pd.read_json(filename)
            z = fp1.var()['z']
            if(z<0.01):
                os.remove(filename)
                print('remove',filename)

def del_dgamma(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            fp1 = pd.read_json(filename)
            z = fp1.var()['alpha']
            if(z<10):
                os.remove(filename)
                print('remove',filename)


if __name__ == "__main__":
    file_path = 'D:\data'
    path1 = 'D:\data\accelerometer'
    path2 = 'D:\data\gyroscope'
    path3 = 'D:\data\device_motion'
    del_size(file_path)
    del_size(file_path)
    del_agz(path1)
    del_agz(path2)
    del_dgamma(path3)

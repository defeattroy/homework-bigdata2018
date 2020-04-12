import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir1 = 'accelerometer/anxiety/female/'
var1 = []
files1 = os.listdir(dir1)
for file in files1:
    path = dir1 + file
    df1 = pd.read_json(path)
    if any(df1.isnull().sum()):
        df1.dropna(how='any')
    variance = np.var(df1.iloc[:,:])
    if variance[0]<0.005 or variance[0]>0.03:
        os.remove(path)
        continue
    var1.append(variance)

dir2 = 'accelerometer/health/female/'
var2 = []
files2 = os.listdir(dir2)
for file in files2:
    path = dir2 + file
    df2 = pd.read_json(path)
    if any(df2.isnull().sum()):
        df2.dropna(how='any')
    variance = np.var(df2.iloc[:,:])
    if variance[0]<0.005 or variance[0]>0.02:
        os.remove(path)
        continue
    var2.append(variance)

dir3 = 'device_motion/anxiety/female/'
var3 = []
files3 = os.listdir(dir3)
for file in files3:
    path = dir3 + file
    df3 = pd.read_json(path)
    if any(df3.isnull().sum()):
        df3.dropna(how='any')
    variance = np.var(df3.iloc[:,:])
    if variance[0]<300 or variance[0]>20000:
        os.remove(path)
        continue
    var3.append(variance)

dir4 = 'device_motion/health/female/'
var4 = []
files4 = os.listdir(dir4)
for file in files4:
    path = dir4 + file
    df4 = pd.read_json(path)
    if any(df4.isnull().sum()):
        df4.dropna(how='any')
    variance = np.var(df4.iloc[:,:])
    if variance[0]<300 or variance[0]>20000:
        os.remove(path)
        continue
    var4.append(variance)

dir5 = 'gyroscope/anxiety/female/'
files5 = os.listdir(dir5)
var5 = []
for file in files5:
    path = dir5 + file
    df5 = pd.read_json(path)
    if any(df5.isnull().sum()):
        df5.dropna(how='any')
    variance = np.var(df5.iloc[:,:])
    if variance[0]<0.008:
        os.remove(path)
        continue
    var5.append(variance)

dir6 = 'gyroscope/health/female/'
files6 = os.listdir(dir6)
var6 = []
count=0
for file in files6:
    path = dir6 + file
    df6 = pd.read_json(path)
    if any(df6.isnull().sum()):
        df6.dropna(how='any')
    variance = np.var(df6.iloc[:,:])
    if variance[1] > 0.1:
        os.remove(path)
        continue
    var6.append(variance)

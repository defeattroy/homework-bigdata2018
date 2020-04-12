import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

"""
预处理步骤：
1.找到并处理空值
2.查看离群点
3.根据答卷时间去除异常值
4.判断手机是否放在桌上，这样的数据不准确，可以考虑去除
5.判断站坐情况的改变，这样的数据不准确，可以考虑去除
"""

def load_data(path):
    data = []
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file)) as f:
                tmp = pd.read_json(f, orient='values')
                data.append(tmp)
    return data


def time_bar_plot(time_anxiety, time_health, name):
    # Bar plot for time_health and time_anxiety
    plt.figure(figsize=(12, 12), dpi=80)
    plt.subplot(2, 1, 1)
    n = len(time_anxiety)
    index = np.arange(n)
    width = 0.45
    p1 = plt.bar(index, time_anxiety, width, label="num", color="#87CEFA")
    plt.xlabel('Number'), plt.ylabel('Time/min'), plt.title(name + ' anxiety')
    plt.yticks(np.arange(0, 150, 5))
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    n = len(time_health)
    index = np.arange(n)
    width = 0.45
    p1 = plt.bar(index, time_health, width, label="num", color="#87CEFA")
    plt.xlabel('Number'), plt.ylabel('Time/min'), plt.title(name + ' health')
    plt.yticks(np.arange(0, 150, 5))
    plt.legend(loc="upper right")
    plt.show()


def domathoftime(anxiety, health):
    # acc_anxiety
    time_anxiety = []
    for _ in anxiety:
        time_anxiety.append((len(_) / 5) / 60)
    # acc_health
    time_health = []
    for _ in health:
        time_health.append((len(_) / 5) / 60)
    return time_anxiety, time_health


def bar_xyz(ax, ay, az, name):
    # Bar plot for abs-mean
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(3, 1, 1)
    n = len(ax)
    index = np.arange(n)
    width = 0.45
    p1 = plt.bar(index, ax, width, label="num", color="#87CEFA")
    plt.xlabel('Number'), plt.ylabel('X'), plt.title(name+'_X')

    plt.subplot(3, 1, 2)
    n = len(ay)
    index = np.arange(n)
    width = 0.45
    p1 = plt.bar(index, ay, width, label="num", color="#87CEFA")
    plt.xlabel('Number'), plt.ylabel('y'), plt.title(name+'_Y')

    plt.subplot(3, 1, 3)
    n = len(az)
    index = np.arange(n)
    width = 0.45
    p1 = plt.bar(index, az, width, label="num", color="#87CEFA")
    plt.xlabel('Number'), plt.ylabel('Z'), plt.title(name+'_Z')

    plt.show()


# Load data
acc_anxiety = load_data(r'D:\work\accelerometer\anxiety\female')
acc_health = load_data(r'D:\work\accelerometer\health\female')
DM_anxiety = load_data(r'D:\work\device_motion\anxiety\female')
DM_health = load_data(r'D:\work\device_motion\health\female')
Gry_anxiety = load_data(r'D:\work\gyroscope\anxiety\female')
Gry_health = load_data(r'D:\work\gyroscope\health\female')
data = [acc_anxiety, acc_health, DM_anxiety, DM_health, Gry_anxiety, Gry_health]



# Do the math of time
time_acc_anxiety, time_acc_health = domathoftime(acc_anxiety, acc_health)
time_DM_anxiety, time_DM_health = domathoftime(DM_anxiety, DM_health)
time_Gry_anxiety, time_Gry_health = domathoftime(Gry_anxiety, Gry_health)

time_bar_plot(time_Gry_anxiety, time_Gry_health, 'Gry')
# It can be seen from the bar plot that there are some missing values in gyroscope data
# because there is no better way to complete it, so delete it

for position in reversed(range(len(Gry_anxiety))):
    if Gry_anxiety[position].empty:
        del Gry_anxiety[position]
for position in reversed(range(len(Gry_health))):
    if Gry_health[position].empty:
        del Gry_health[position]
time_Gry_anxiety, time_Gry_health = domathoftime(Gry_anxiety, Gry_health)


# 离散值、这里只画了一点，可以都画出来看看，为了不产生太多的图我没有写在这里
plt.boxplot(acc_anxiety[0]['x'].tolist())
plt.show()


# bar plot for accelerometer device_motion and gyroscope
time_bar_plot(time_acc_anxiety, time_acc_health, 'ACC')
time_bar_plot(time_DM_anxiety, time_DM_health, 'DM')
time_bar_plot(time_Gry_anxiety, time_Gry_health, 'Gry')

# Remove number that use too long or short time
for _ in data:
    for position in reversed(range(len(_))):
        if len(_[position])/300 > 100 or len(_[position]/300) < 10:
            del _[position]


# 根据X,Y,Z三个坐标的方差、绝对值后的平均值，这两种数据分别画图，检查出那些放在桌子上的数据组。
# 若方差过小，可能是手机放在桌上。 若绝对值后的平均值仍然趋近于0，可能放在桌子上。
name = ['acc_anxiety', 'acc_health', 'DM_anxiety', 'DM_health', 'Gry_anxiety', 'Gry_health']
for pos in range(len(data)):
    a = []
    for i in data[pos]:
        ax, ay, az = [], [], []
        tmp = (i.abs().mean()).tolist()
        a.append(tmp)
        for x in a:
            ax.append(x[0])
            ay.append(x[1])
            az.append(x[2])
    bar_xyz(ax, ay, az, name[pos])

for pos in range(len(data)):
    a = []
    for i in data[pos]:
        ax, ay, az = [], [], []
        tmp = (i.var()).tolist()
        a.append(tmp)
        for x in a:
            ax.append(x[0])
            ay.append(x[1])
            az.append(x[2])
    bar_xyz(ax, ay, az, name[pos])

# var方差
for pos in range(len(data)):
    a = []
    for i in data[pos]:
        ax, ay, az = [], [], []
        tmp = (i.var()).tolist()
        a.append(tmp)
        for x in a:
            ax.append(x[0])
            ay.append(x[1])
            az.append(x[2])
    bar_xyz(ax, ay, az, name[pos])


# 取三个坐标前30% 和后70%的方差，计算其差值，差值过大的，可能是出现了站立和坐下的切换，这样的数据不准确，可以考虑去除。
for pos in range(len(data[:2])):
    for d in data[pos]:
        x_sub = d[d.columns[0]][:int(d.shape[0]*0.3)].var()-d[d.columns[0]][int(d.shape[0]*0.3):].var()
        y_sub = d[d.columns[1]][:int(d.shape[0]*0.3)].var()-d[d.columns[1]][int(d.shape[0]*0.3):].var()
        z_sub = d[d.columns[2]][:int(d.shape[0]*0.3)].var()-d[d.columns[2]][int(d.shape[0]*0.3):].var()
        if x_sub < 0.001 or y_sub < 0.001 or z_sub < 0.001:
            # 出问题的数据
            print(x_sub, y_sub, z_sub)
            #d.plot()


# 删除
# delete nosiy data
del_list = [10, 23]
cnt = 0
for x in del_list:
    del acc_anxiety[x - cnt]
    cnt += 1

del_list = [16, 26]
cnt = 0
for x in del_list:
    del acc_health[x - cnt]
    cnt += 1

del_list = [10, 4, 16, 23, 29]
del_list.sort()
cnt = 0
for x in del_list:
    del DM_anxiety[x - cnt]
    cnt += 1

del_list = [12, 15, 19]
cnt = 0
for x in del_list:
    del DM_health[x - cnt]
    cnt += 1

del_list = [8]
cnt = 0
for x in del_list:
    del Gry_anxiety[x - cnt]
    del Gry_health[x - cnt]
    cnt += 1


# 写回数据，还没有完全了解陀螺仪的数据怎么分析，所以我认为还没有做完分析工作，故还没编写写回的代码
pass

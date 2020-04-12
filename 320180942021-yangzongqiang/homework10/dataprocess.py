import pandas as pd
import matplotlib.pyplot as plt
import os


"""
数据预处理部分：
1.找出空文件进行删除
2.寻找空值
3.寻找离群点
4.计算答题所用时间
5.判断手机是否放在桌上
6.判断站坐情况是否发生改变
"""
filename = input("请输入要处理的文件名:")
# 判断文件是否存在
if filename not in os.listdir('.'):
    print("文件不存在")
    exit(0)

print(filename+"处理中")

# 查看是否为空文件或数据大小过小
if os.path.getsize(filename) < 10:
    print("文件过小，已进行删除")
    os.remove(filename)
    exit(0)

# 读取数据
with open(filename, 'r') as f:
    _ = f.read()
# 数据转换
data = pd.read_json(_)

# 展示出各坐标轴空值总和
print(data.isnull().sum())

if data.isnull().sum().sum() != 0:
    print("存在空值，需要进行处理")

# 画出各坐标轴盒图，寻找特殊坐标
plt.subplot(131)
plt.boxplot(data[data.columns[0]])
plt.subplot(132)
plt.boxplot(data[data.columns[1]])
plt.subplot(133)
plt.boxplot(data[data.columns[2]])
plt.show()

# 计算该问卷数据回答时间，初步判断是否为有效数据
answer_time = data.shape[0]/5/60
if answer_time > 90 or answer_time < 10:
    print("答题时长异常（少于10min或大于90min），建议删除该数据")

# 计算该问卷每个坐标轴方差，若方差过小，可能是手机放在桌上，未拿在手上
if data[data.columns[0]].var() < 0.001 or data[data.columns[1]].var() < 0.001 or data[data.columns[2]].var() < 0.001:
    print("可能手机放在平面，未拿在手上，建议删除")

# 计算前25%数据与后75%数据方差之差，若差值过大，可能是站起来（或坐下去）了
x_var_sub = data[data.columns[0]][:int(data.shape[0]*0.25)].var()-data[data.columns[0]][int(data.shape[0]*0.25):].var()
y_var_sub = data[data.columns[1]][:int(data.shape[0]*0.25)].var()-data[data.columns[1]][int(data.shape[0]*0.25):].var()
z_var_sub = data[data.columns[2]][:int(data.shape[0]*0.25)].var()-data[data.columns[2]][int(data.shape[0]*0.25):].var()
if x_var_sub < 0.001 or y_var_sub < 0.001 or z_var_sub < 0.001:
    print("可能先站后坐（先坐后站），数据参考价值不高")
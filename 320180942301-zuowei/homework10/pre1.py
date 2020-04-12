import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  
import json 
import pandas as pd
import os

#问卷时间分布柱状图
dir_data='./data/'
xaxis=['time = 0', '0 < time <= 10', '10 < time <= 20', '20 < time <= 30','30 < time <= 40', '40 < time <= 50',
       '50 < time <= 60', '60 < time <= 70', '70 < time <= 80', '80 < time <= 90', '90 < time <= 100', 'time >= 100']
time_dic={}.fromkeys(xaxis, 0) 
for root, dirs, files in os.walk(dir_data):
	for the_file in files:
		path = os.path.join(root,the_file)
		with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				length = len(data)
				time=round(length / 5 / 60, 2) #收集频率为5Hz
				if time == 0: time_dic['time = 0']+=1
				elif 0 < time <=10: time_dic['0 < time <= 10']+=1
				elif 10 < time <= 20: time_dic['10 < time <= 20']+=1
				elif 20 < time <= 30: time_dic['20 < time <= 30']+=1
				elif 30 < time <= 40: time_dic['30 < time <= 40']+=1 
				elif 40 < time <= 50: time_dic['40 < time <= 50']+=1
				elif 50 < time <= 60: time_dic['50 < time <= 60']+=1
				elif 60 < time <= 70: time_dic['60 < time <= 70']+=1
				elif 70 < time <= 80: time_dic['70 < time <= 80']+=1
				elif 80 < time <= 90: time_dic['80 < time <= 90']+=1
				elif 90 < time <= 100: time_dic['90 < time <= 100']+=1 
				else: time_dic['time >= 100']+=1
				
fig=plt.figure(figsize=(19,9))
plt.bar(list(time_dic.keys()), list(time_dic.values()))
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
plt.xlabel('问卷所用时间（单位：分钟）', FontProperties=font)
plt.ylabel('问卷数', FontProperties=font)
plt.title('问卷时间分布图', FontProperties=font)
plt.xticks(rotation=20,fontsize=10) 
plt.savefig('./问卷时间分布图.jpg')

#accelerometer分析
df1=pd.read_json("./data/accelerometer/anxiety/female/20191107191800_308_accelerometer.json")
ax1=df1.plot()
fig=ax1.get_figure()
fig.savefig('./accelerometer走势示例1.jpg')
print('accelerometer示例1方差结果')
print(df1.iloc[500:].var())

df2=pd.read_json("./data/accelerometer/anxiety/female/20191110161313_1383_accelerometer.json")
ax2=df2.plot()
fig=ax2.get_figure()
fig.savefig('./accelerometer走势示例2.jpg')
print('accelerometer示例2方差结果')
print(df2.iloc[500:].var())

#device_motion分析
df3=pd.read_json("./data/device_motion/anxiety/female/20191110161313_1383_device_motion.json")
ax3=df3.plot()
fig=ax3.get_figure()
fig.savefig('./device_motion走势示例1.jpg')
print('device_motion示例1方差结果')
print(df3.iloc[500:].var())

df4=pd.read_json("./data/device_motion/anxiety/female/20191109190044_1100_device_motion.json")
ax4=df4.plot()
fig=ax4.get_figure()
fig.savefig('./device_motion走势示例2.jpg')
print('device_motion示例2方差结果')
print(df4.iloc[500:].var())

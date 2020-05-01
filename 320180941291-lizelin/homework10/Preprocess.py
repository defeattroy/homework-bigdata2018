import pandas as pd 
import json 
import os

dir_acc = "./data/accelerometer/"
dir_dm = "./data/device_motion/"
dir_gy = "./data/gyroscope/"
dir_aa="./preprocessed_data/accelerometer/anxiety/female/"
dir_ah="./preprocessed_data/accelerometer/health/female/"
dir_da="./preprocessed_data/device_motion/anxiety/female/"
dir_dh="./preprocessed_data/device_motion/health/female/"
dir_ga="./preprocessed_data/gyroscope/anxiety/female/"
dir_gh="./preprocessed_data/gyroscope/health/female/"

for path in [dir_aa, dir_ah, dir_da, dir_dh, dir_ga, dir_gh]: #创建处理结果文件夹
	if os.path.exists(path)==False:
            os.makedirs(path)
#处理accelerometer数据
for root, dirs, files in os.walk(dir_acc):
	for the_file in files:
		path = os.path.join(root,the_file)
		with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				length = len(data)
				time = round(length / 5 / 60, 2) #获取时间
				if 10<time<=100:                           #剔除时间过长或过短的文件
					df=pd.read_json(path,encoding='utf-8')
					df = df.iloc[500:]
					var = df['x'].var()                    #计算x轴的方差
					if var > 0.001:                        #剔除方差过小的无效数据
						df.to_json("./preprocessed_data"+path[6:], orient='records')         #保存文件
					
					
#处理device_motion数据
for root, dirs, files in os.walk(dir_dm):
	for the_file in files:
		path = os.path.join(root,the_file)
		with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				length = len(data)
				time = round(length / 5 / 60, 2) #获取时间
				if 10<time<=100:                        #剔除时间过长或过短的文件
					df=pd.read_json(path,encoding='utf-8')
					df = df.iloc[500:]
					var = df['alpha'].var()            #计算alpha的方差值
					if var > 3:                        #剔除方差过小的无效数据
						df.to_json("./preprocessed_data"+path[6:], orient='records')         #保存文件

#处理gyroscope数据						
for root, dirs, files in os.walk(dir_gy):
	for the_file in files:
		path = os.path.join(root,the_file)
		with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				length = len(data)
				time = round(length / 5 / 60, 2) #获取时间
				if 10<time<=100:                    #剔除空文件
					df=pd.read_json(path,encoding='utf-8')
					df = df.iloc[500:]
					df.to_json("./preprocessed_data"+path[6:], orient='records')         #保存文件


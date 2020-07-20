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

for path in [dir_aa, dir_ah, dir_da, dir_dh, dir_ga, dir_gh]: #�����������ļ���
	if os.path.exists(path)==False:
            os.makedirs(path)
#����accelerometer����
for root, dirs, files in os.walk(dir_acc):
	for the_file in files:
		path = os.path.join(root,the_file)
		with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				length = len(data)
				time = round(length / 5 / 60, 2) #��ȡʱ��
				if 10<time<=100:                           #�޳�ʱ���������̵��ļ�
					df=pd.read_json(path,encoding='utf-8')
					df = df.iloc[500:]
					var = df['x'].var()                    #����x��ķ���
					if var > 0.001:                        #�޳������С����Ч����
						df.to_json("./preprocessed_data"+path[6:], orient='records')         #�����ļ�
					
					
#����device_motion����
for root, dirs, files in os.walk(dir_dm):
	for the_file in files:
		path = os.path.join(root,the_file)
		with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				length = len(data)
				time = round(length / 5 / 60, 2) #��ȡʱ��
				if 10<time<=100:                        #�޳�ʱ���������̵��ļ�
					df=pd.read_json(path,encoding='utf-8')
					df = df.iloc[500:]
					var = df['alpha'].var()            #����alpha�ķ���ֵ
					if var > 3:                        #�޳������С����Ч����
						df.to_json("./preprocessed_data"+path[6:], orient='records')         #�����ļ�

#����gyroscope����						
for root, dirs, files in os.walk(dir_gy):
	for the_file in files:
		path = os.path.join(root,the_file)
		with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				length = len(data)
				time = round(length / 5 / 60, 2) #��ȡʱ��
				if 10<time<=100:                    #�޳����ļ�
					df=pd.read_json(path,encoding='utf-8')
					df = df.iloc[500:]
					df.to_json("./preprocessed_data"+path[6:], orient='records')         #�����ļ�

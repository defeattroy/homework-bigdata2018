import os
import pandas as pd
import matplotlib.pyplot as plt

dir = "/Users/huyelong/Desktop/homework10/data/accelerometer/anxiety/female/"
for root, dirs, files in os.walk(dir):
	for file in files:
		path = os.path.join(root,file)
		#print(path) <-checkpoint
		if path[-5:] == ".json":
			with open(path, 'r', encoding = 'utf-8') as f:
				data = pd.read_json(f)
				print(data.isnull())   #查看data是否有缺失值（空值），False无缺失值
				data.dropna()	#删除带空值的行
				print(data.describe())
				data.plot.box(title = file)
				plt.grid(linestyle = "--", alpha = 0.3)
				plt.show()
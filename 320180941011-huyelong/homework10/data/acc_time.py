import json
import os
from pyecharts import options as opts
from pyecharts.charts import Bar

counts_aa = 0
counts_ah = 0
time_aa = 0
time_ah = 0

aa_yaxis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ah_yaxis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#time is 0~10,10~20,20~30,30~40,40~50,50~60,60~70,70~80,80~90,90~100, over 100 mins

dir1 = "/Users/huyelong/Desktop/homework10/data/accelerometer/anxiety/female/"
for root, dirs, files in os.walk(dir1):
	for file in files:
		path = os.path.join(root,file)
		#print(path) <-checkpoint
		if path[-5:] == ".json":
			with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				counts_aa = len(data)
				time_aa = round(counts_aa / 5 / 60, 2) #sample frequency 5Hz
				#print(time_aa) <-checkpoint
				if time_aa <= 0:
					pass
				elif 0 < time_aa <=10:
					aa_yaxis[0] += 1
				elif 10 < time_aa <= 20:
					aa_yaxis[1] += 1
				elif 20 < time_aa <= 30:
					aa_yaxis[2] += 1
				elif 30 < time_aa <= 40:
					aa_yaxis[3] += 1
				elif 40 < time_aa <= 50:
					aa_yaxis[4] += 1
				elif 50 < time_aa <= 60:
					aa_yaxis[5] += 1
				elif 60 < time_aa <= 70:
					aa_yaxis[6] += 1
				elif 70 < time_aa <= 80:
					aa_yaxis[7] += 1
				elif 80 < time_aa <= 90:
					aa_yaxis[8] += 1
				elif 90 < time_aa <= 100:
					aa_yaxis[9] += 1
				else:
					aa_yaxis[10] += 1
#print(aa_yaxis) <-checkpoint


dir2 = "/Users/huyelong/Desktop/homework10/data/accelerometer/health/female/"
for root, dirs, files in os.walk(dir2):
	for file in files:
		path = os.path.join(root,file)
		#print(path) <-checkpoint
		if path[-5:] == ".json":
			with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				counts_ah = len(data)
				time_ah = round(counts_ah / 5 / 60, 2) #sample frequency 5Hz
				#print(time_ah) <-checkpoint
				if time_ah <= 0:
					pass
				elif 0 < time_ah <=10:
					ah_yaxis[0] += 1
				elif 10 < time_ah <= 20:
					ah_yaxis[1] += 1
				elif 20 < time_ah <= 30:
					ah_yaxis[2] += 1
				elif 30 < time_ah <= 40:
					ah_yaxis[3] += 1
				elif 40 < time_ah <= 50:
					ah_yaxis[4] += 1
				elif 50 < time_ah <= 60:
					ah_yaxis[5] += 1
				elif 60 < time_ah <= 70:
					ah_yaxis[6] += 1
				elif 70 < time_ah <= 80:
					ah_yaxis[7] += 1
				elif 80 < time_ah <= 90:
					ah_yaxis[8] += 1
				elif 90 < time_ah <= 100:
					ah_yaxis[9] += 1
				else:
					ah_yaxis[10] += 1
#print(ah_yaxis) <-checkpoint

bar = (
	Bar()
	.add_xaxis(
		[
			"0到10分钟",
			"10到20分钟",
			"20到30分钟",
			"30到40分钟",
			"40到50分钟",
			"50到60分钟",
			"60到70分钟",
			"70到80分钟",
			"80到90分钟",
			"90到100分钟",
			"超过100分钟"
		]
	)
	.add_yaxis("anxiety", aa_yaxis)
	.add_yaxis("health", ah_yaxis)
	.set_global_opts(
		xaxis_opts = opts.AxisOpts(axislabel_opts = opts.LabelOpts(rotate = -20)),
        title_opts = opts.TitleOpts(title = "加速度计-测试时间对比图", subtitle = "anxiety群组和health群组回答问卷所用时间对比柱状图"),
	)
	.render("acc_time_contrast.html")
)
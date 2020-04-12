import json
import os
from pyecharts.charts import Scatter3D
import pyecharts.options as opts

dir = "/Users/huyelong/Desktop/homework10/data/device_motion/anxiety/female/"
for root, dirs, files in os.walk(dir):
	for file in files:
		path = os.path.join(root,file)
		print(path)
		if path[-5:] == ".json":
			with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)

				volume = [[i['alpha'], i['beta'],i['gamma']] for i in data]

				range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
       				       	 '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
				scatter3D = (
					Scatter3D(init_opts = opts.InitOpts(width = "1440px", height = "720px"))
					.add("Sensor Data", volume)
					.set_global_opts(
						title_opts = opts.TitleOpts(title = file),
						visualmap_opts = opts.VisualMapOpts(is_show = True, max_ = max(i['gamma'] for i in data), min_ = min(i['gamma'] for i in data), range_color = range_color)
						)
					.render(path[:-5] + ".html")
				)


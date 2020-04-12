import json
import os
from pyecharts.charts import Line
import pyecharts.options as opts

dir = "/Users/huyelong/Desktop/homework10/data/accelerometer/health/female/"
for root, dirs, files in os.walk(dir):
	for file in files:
		path = os.path.join(root,file)
		print(path)
		if path[-5:] == ".json":
			with open(path, 'r', encoding = 'utf-8') as f:
				data = json.load(f)
				line = (
					Line(init_opts = opts.InitOpts(width = "1440px", height = "720px"))
					.add_xaxis(list(range(1001)))
					.add_yaxis("x轴加速度", [i['x'] for i in data])
					.add_yaxis("y轴加速度", [i['y'] for i in data])
					.add_yaxis("z轴加速度", [i['z'] for i in data])
					.set_series_opts(label_opts = opts.LabelOpts(is_show = False))
					.set_global_opts(title_opts = opts.TitleOpts(title = file))
					.render(file[:-5] + ".html")
				)
import requests

#读取网页数据,读取每一个json文件的数据
url="http://yang.lzu.edu.cn/data/accelerometer/anxiety/female/20191108111045_558_accelerometer.json"
r = requests.get(url)
r.raise_for_status()
r.encoding=r.apparent_encoding
#将网页数据保存到对应的文件夹的文本文档中
fp = open(r'E:\jupyter-notebook\accelerometer-anxiety-female\08111045_558.json',mode='w')
fp.write(r.text)

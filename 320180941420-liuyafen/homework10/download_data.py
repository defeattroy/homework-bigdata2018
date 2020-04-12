import requests
from bs4 import BeautifulSoup
import re

"""
way1到way6是6个文件夹的路径
要先用http://yang.lzu.edu.cn/data/index.txt的网址爬出文件名
然后用文件名爬取文件到相应的文件夹
"""

way1 = 'C:\\homework10\\accelerometer\\anxiety'
way2 = 'C:\\homework10\\accelerometer\\health'
way3 = 'C:\\homework10\\device_motion\\anxiety'
way4 = 'C:\\homework10\\device_motion\\health'
way5 = 'C:\\homework10\\gyroscope\\anxiety'
way6 = 'C:\\homework10\\\gyroscope\\health'
#爬取文件
url='http://yang.lzu.edu.cn/data/index.txt'
data=requests.get(url)
soup=BeautifulSoup(data.text,'lxml')  #解析

name = []
r = str(soup)
r = r.split('\n')


for line in r:
    if '.json' in line and line.startswith('./'):
        a = re.findall(r"./(.*)", line)
        name.append(a)


for i in name:
    a = (str(i))
    a = a[2:-2]
    url = 'http://yang.lzu.edu.cn/data/' + a
    d = requests.get(url)
    d.raise_for_status()
    b = a.replace('/','_')  #文件名
    if 'accelerometer_anxiety' in b:
        fw=open(way1 + '\\' + b,'w',encoding='utf-8')
        fw.write(d.text)
        fw.close()
    elif 'accelerometer_health' in b:
        fw=open(way2 + '\\' + b,'w',encoding='utf-8')
        fw.write(d.text)
        fw.close()
    elif 'device_motion_anxiety' in b:
        fw=open(way3 + '\\' + b,'w',encoding='utf-8')
        fw.write(d.text)
        fw.close()
    elif 'device_motion_health' in b:
        fw=open(way4 + '\\' + b,'w',encoding='utf-8')
        fw.write(d.text)
        fw.close()
    elif 'gyroscope_anxiety' in b:
        fw=open(way5 + '\\' + b,'w',encoding='utf-8')
        fw.write(d.text)
        fw.close()
    elif 'gyroscope_health' in b:
        fw=open(way6 + '\\' + b,'w',encoding='utf-8')
        fw.write(d.text)
        fw.close()

import requests
import re
import os
import time
#简单的爬虫程序，爬完之后我手工分类到了文件夹
url = 'http://yang.lzu.edu.cn/data/index.txt'
respons = requests.get(url)
respons.encoding = 'utf-8'
spawn_html = respons.text
spawn_html = spawn_html.split()
for x in spawn_html:
    if(x.endswith('.json') and x.startswith('./gyroscope')):
        new_url = 'http://yang.lzu.edu.cn/data' + x.lstrip('.')
        #print(new_url)
        new_respons = requests.get(new_url)
        txt = new_respons.text
        x = x.replace('/', '.')
        x = x.lstrip('.')
        #print(x)
        path = r'D:\work'
        with open(path + './' + x, 'w') as fd:
            fd.write(txt)
            print("finish" + x)


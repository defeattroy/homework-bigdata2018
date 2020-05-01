import json
import requests
import urllib.request
import os

def get_text(url):  #爬取json数据
    resp = urllib.request.urlopen(url)
    ele_json = json.loads(resp.read())
    return ele_json

def make_dir(path_list): #创建数据对应文件夹
    for x in path_list:
        path='./data'+x[1:]
        if os.path.exists(x)==False:
            os.makedirs(path)
            
def download_json(path): #将json数据保存在本地
	try:
		full_path='http://yang.lzu.edu.cn/data'+path[1:]
		ele_json=get_text(full_path)
		file_path='./data'+path[1:]
		with open(file_path, 'w') as f:
			json.dump(ele_json, f)
	except urllib.error.HTTPError as e:
		print('文件<'+path+'>下载失败')
		
        
url='http://yang.lzu.edu.cn/data/index.txt'
res=requests.get(url)
res.encoding='utf-8'
page=res.text
sum_list=page.split('\n')
path_list=[x+'/' for x in sum_list if x.endswith('.json')==False and x.startswith('./')]
make_dir(path_list)
json_list=[x for x in sum_list if x.endswith('.json')]
for path in json_list:
	download_json(path)
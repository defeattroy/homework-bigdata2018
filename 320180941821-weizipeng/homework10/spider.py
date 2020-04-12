# 爬取数据
import requests
import os
r = requests.get('http://yang.lzu.edu.cn/data/index.txt')
# 爬取数据的处理
content = r.text.splitlines(False) #按换行符分割
paths = []
file_path = r'D:\Python练习\homework10\sourcedata' # 爬取数据的本地存储路径
for path in content: # 去掉开头的‘./’,保留完整文件路径的字符串,文件夹路径用来创建本地文件夹
    path = path.lstrip('./')
    if path.endswith("json"):
        paths.append(path)
    else:
        os.makedirs(os.path.join(file_path,path))

urls = list(map(lambda x: r'http://yang.lzu.edu.cn/data/'+x,paths)) # 获取完整url
url_files = dict(zip(urls,paths))
for url , path in url_files.items():
    try:
        r = requests.get(url)
    except:
        print('访问失败,file_name未保存')
    with open(os.path.join(file_path,path),'w') as f:
        f.write(r.text)
        f.close()
        print(path.split('/')[-1]+'已保存')
print('已完成')
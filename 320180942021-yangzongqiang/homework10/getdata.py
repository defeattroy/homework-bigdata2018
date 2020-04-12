import requests


"""
获取数据部分：
1.使用request获取index.txt页面下所有的路径
2.构造路径和文件名
3.请求每一条数据并保存到本地
"""
# 请求地址首部
url = "http://yang.lzu.edu.cn/data/"
# 伪造chrome浏览器请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                  ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
}
# 拿到所有文件路径
r = requests.get(url+"index.txt", headers=headers)

_ = r.text.replace('./', '').split('\n')[:-1]  # 按换行符分割，将./删除
new_url = [i for i in _ if len(i) > 40]  # 构造出所有数据的url
new_name = [i.replace('/', '_') for i in new_url]  # 构造出所有数据的文件名

# 拿到所有数据并保存
for i in range(len(new_url)):
    # 读取数据
    r = requests.get(url + new_url[i], headers=headers)
    with open(new_name[i], 'w') as f:
        f.write(r.text)
        print(new_name[i]+"保存成功")

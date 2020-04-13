import requests,os,sys

url = "http://yang.lzu.edu.cn/data/"
# 伪造chrome浏览器请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                  ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',

}

# 拿到所有文件路径

r = requests.get(url+"index.txt", headers=headers)
_ = r.text.replace('./','').split('\n')[:-1]  # 按换行符分割，将./删除
new_url = [i for i in _ if len(i) > 40]  # 构造出所有数据的url
new_name = [i.replace('/', '_') for i in new_url]  # 构造出所有数据的文件名

#创建目标存储目录
os.mkdir('D:\data')
path="/tmp"
# 查看当前工作目录
retval = os.getcwd()
print("当前工作目录为%s"%retval)
os.chdir('D:\data')
retval=os.getcwd()
print("目录修改成功%s"%retval)

for i in range(len(new_url)):
    # 读取数据
    r = requests.get(url + new_url[i], headers=headers)
    with open(new_name[i], 'w') as f:
        f.write(r.text)
        print(new_name[i]+"保存成功")
'''_device_motion_health_female_20191114174238_3045_device_motion文件
由于下载目录名称不完整，需手动修改名称'''

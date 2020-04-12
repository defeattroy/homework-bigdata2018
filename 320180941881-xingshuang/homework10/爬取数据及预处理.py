import requests
import pandas as pd

#选择要爬的网址
url = 'http://yang.lzu.edu.cn/data/index.txt'
response = requests.get(url)
response.encoding = 'utf-8'

#获取网页源码
html = response.text

data=html.split('\n')
data.remove("./accelerometer/anxiety/female")
data.remove("./accelerometer/health/female")
data.remove("./device_motion/anxiety/female")
data.remove("./device_motion/health/female")
data.remove("./gyroscope/anxiety/female")
data.remove("./gyroscope/health/female")
data.pop()
print(data)
'''
data列表：
accelerometer：0 -- 77
device_motion：78 -- 144
gyroscope： 145 -- 198
'''

'''
将爬到的数据保存在当前目录下 “ 原数据 ” 文件夹中
'''
for fn in data:
    url_2 = "http://yang.lzu.edu.cn/data" + fn.strip('.')
    response_2 = requests.get(url_2)
    response_2.encoding = 'utf-8'
    html_2 = response_2.text

    fname=fn.strip('./').replace('/','_')
    print(fname)

    with open("./原数据/"+fname,"w") as fp:
        fp.write(html_2)


'''
从原数据文件夹中读取数据，分类进行预处理，与处理完成后写入当前目录下的 “ 预处理后数据 ” 文件夹中。
'''
for fn in data[0:78]:                           # accelerometer
    fname=fn.strip('./').replace('/','_')
    df=pd.read_json("./原数据/"+fname)

    if df.empty:                                #  计算答题时间
        time = 0
    else:
        time = df.iloc[:,0].size // (5*60)
    if 8 < time < 60:                          #  剔除掉不在合理时间范围内的数据
        df = df.iloc[500:]                     #  去除  “前几分钟未开始答题 ”  的无效数据
        var = df['x'].var()                    #  计算 x 轴 的方差
        if var > 0.001:                        #  根据方差剔除掉  “将手机置于桌子上” 的无效数据
            df.to_json("./预处理后数据/"+fname, orient='records')         #  写入文件

for fn in data[78:145]:                        # device_motion
    fname=fn.strip('./').replace('/','_')
    df=pd.read_json("./原数据/"+fname)

    if df.empty:
        time = 0
    else:
        time = df.iloc[:,0].size // (5*60)
    if 8 < time < 60:
        df = df.iloc[500:]
        var = df['alpha'].var()
        if var > 3:
            df.to_json("./预处理后数据/"+fname, orient='records')

for fn in data[145:199]:                        #  gyroscope
    fname=fn.strip('./').replace('/','_')
    df=pd.read_json("./原数据/"+fname)

    if df.empty:
        time = 0
    else:
        time = df.iloc[:,0].size // (5*60)
    if 8 < time < 60:
        df = df.iloc[500:]
        df.to_json("./预处理后数据/"+fname, orient='records')



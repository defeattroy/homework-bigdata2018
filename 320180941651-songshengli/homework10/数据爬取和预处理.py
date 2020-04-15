import requests
import pandas as pd
import os

# 选择要爬的网址
url = 'http://yang.lzu.edu.cn/data/index.txt'
response = requests.get(url)
response.encoding = 'utf-8'

# 获取网页源码
html = response.text

data = html.split('\n')
data.remove("./accelerometer/anxiety/female")
data.remove("./accelerometer/health/female")
data.remove("./device_motion/anxiety/female")
data.remove("./device_motion/health/female")
data.remove("./gyroscope/anxiety/female")
data.remove("./gyroscope/health/female")

accelerometer_anxiety_female = []
for i in data:
    i = i.replace("./accelerometer/anxiety/female/", "")
    if len(i) < 50 and len(i) > 0:
        accelerometer_anxiety_female.append(i)
    # print(accelerometer_anxiety_female)

accelerometer_health_female = []            ##获取六种分类中的文件名，放在六个列表
for i in data:
    i = i.replace("./accelerometer/health/female/", "")
    if len(i) < 50 and len(i) > 10:
        accelerometer_health_female.append(i)
    # print(accelerometer_health_female)

device_motion_anxiety_female = []
for i in data:
    i = i.replace("./device_motion/anxiety/female/", "")
    if len(i) < 50 and len(i) > 10:
        device_motion_anxiety_female.append(i)
    # print(device_motion_anxiety_female)

device_motion_health_female = []
for i in data:
    i = i.replace("./device_motion/health/female/", "")
    if len(i) < 50 and len(i) > 10:
        device_motion_health_female.append(i)
    # print(device_motion_health_female)

gyroscope_anxiety_female = []
for i in data:
    i = i.replace("./gyroscope/anxiety/female/", "")
    if len(i) < 50 and len(i) > 10:
        gyroscope_anxiety_female.append(i)
    # print(gyroscope_anxiety_female)

gyroscope_health_female = []
for i in data:
    i = i.replace("./gyroscope/health/female/", "")
    if len(i) < 50 and len(i) > 10:
        gyroscope_health_female.append(i)
    # print(gyroscope_health_female)

for j in accelerometer_anxiety_female:    #  将数据爬下来分别放在各个文件中
    url_2 = "http://yang.lzu.edu.cn/data/accelerometer/anxiety/female/" + j
    response_2 = requests.get(url_2)
    response_2.encoding = 'utf-8'
    html_2 = response_2.text
    with open("./爬取的原数据/accelerometer_anxiety_female/" + j, "w") as fp:
        fp.write(html_2)
    # print(j)

for j in accelerometer_health_female:
    url_2 = "http://yang.lzu.edu.cn/data/accelerometer/health/female/" + j
    response_2 = requests.get(url_2)
    response_2.encoding = 'utf-8'
    html_2 = response_2.text
    with open("./爬取的原数据/accelerometer_health_female/" + j, "w") as fp:
        fp.write(html_2)
    # print(j)

for j in device_motion_anxiety_female:
    url_2 = "http://yang.lzu.edu.cn/data/device_motion/anxiety/female/" + j
    response_2 = requests.get(url_2)
    response_2.encoding = 'utf-8'
    html_2 = response_2.text
    with open("./爬取的原数据/device_motion_anxiety_female/" + j, "w") as fp:
        fp.write(html_2)
    # print(j)

for j in device_motion_health_female:
    url_2 = "http://yang.lzu.edu.cn/data/device_motion/health/female/" + j
    response_2 = requests.get(url_2)
    response_2.encoding = 'utf-8'
    html_2 = response_2.text
    with open("./爬取的原数据/device_motion_health_female/" + j, "w") as fp:
        fp.write(html_2)
    # print(j)

for j in gyroscope_anxiety_female:
    url_2 = "http://yang.lzu.edu.cn/data/gyroscope/anxiety/female/" + j
    response_2 = requests.get(url_2)
    response_2.encoding = 'utf-8'
    html_2 = response_2.text
    with open("./爬取的原数据/gyroscope_anxiety_female/" + j, "w") as fp:
        fp.write(html_2)
    # print(j)

for j in gyroscope_health_female:
    url_2 = "http://yang.lzu.edu.cn/data/gyroscope/health/female/" + j
    response_2 = requests.get(url_2)
    response_2.encoding = 'utf-8'
    html_2 = response_2.text
    with open("./爬取的原数据/gyroscope_health_female/" + j, "w") as fp:
        fp.write(html_2)
    # print(j)

file = os.listdir('爬取的原数据')
for i in file:  # 文件分类的根目录
    list = os.listdir('爬取的原数据/' + i)
    for j in list:  # 读取每一个文件
        data = pd.read_json('爬取的原数据/' + i + '/' + j)
        column = data.columns.values  # 获取列名
        time = data.shape[0] / 5 / 60
        if time >= 10 and time <= 60:  # 保留10到60分钟的文件
            data.fillna(value={column[0]: data[column[0]].mean(),  # 使用平均值替换空值
                               column[1]: data[column[1]].mean(),
                               column[2]: data[column[2]].mean()
                               },
                        inplace=True
                        )
            if not (data[column[0]].var() < 0.001 or
                    data[column[1]].var() < 0.001 or data[
                        column[2]].var() < 0.001):  # 如果某一方向方差过小，可能由于设备故障或者没有拿在手上，不保留改数据

                data.to_json('预处理后的数据/' + i + '_' + j, orient='table')  # 将数据清理好的文件写入(命名格式为分类加时间戳id)
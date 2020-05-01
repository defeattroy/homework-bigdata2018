import requests
import re

#选择爬取的url
url = 'http://yang.lzu.edu.cn/data/index.txt'
r = requests.get(url)
s = str(r.text)
r0 = s.split('\n')

#获取相应数据编号
aaf = []
ahf = []
daf = []
dhf = []
gaf = []
ghf = []
for line in r0:
    if line.startswith('./accelerometer/anxiety/female')and line.endswith('.json'):
        a = re.findall(r'female/(.*)_accelerometer', line)
        aaf.append(str(a))
    elif line.startswith('./accelerometer/health/female')and line.endswith('.json'):
        a = re.findall(r'female/(.*)_accelerometer', line)
        ahf.append(str(a))
    elif line.startswith('./device_motion/anxiety/female')and line.endswith('.json'):
        a = re.findall(r'female/(.*)_device_motion', line)
        daf.append(str(a))
    elif line.startswith('./device_motion/health/female')and line.endswith('.json'):
        a = re.findall(r'female/(.*)_device_motion', line)
        dhf.append(str(a))
    elif line.startswith('./gyroscope/anxiety/female')and line.endswith('.json'):
        a = re.findall(r'female/(.*)_gyroscope', line)
        gaf.append(str(a))
    elif line.startswith('./gyroscope/health/female')and line.endswith('.json'):
        a = re.findall(r'female/(.*)_gyroscope', line)
        ghf.append(str(a))

#保存相应数据到本地
for i in aaf:
    url1 = 'http://yang.lzu.edu.cn/data/accelerometer/anxiety/female/'+i.strip()[2:-2]+'_accelerometer.json'
    r1 = requests.get(url1)
    fp1 = open(r'D:\data\accelerometer\anxiety\female\{}_accelerometer.json'.format(i.strip()[2:-2]),mode='w')
    fp1.write(r1.text)
fp1.close()

for i in ahf:
    url2 = 'http://yang.lzu.edu.cn/data/accelerometer/health/female/'+i.strip()[2:-2]+'_accelerometer.json'
    r2 = requests.get(url2)
    fp2 = open(r'D:\data\accelerometer\health\female\{}_accelerometer.json'.format(i.strip()[2:-2]),mode='w')
    fp2.write(r2.text)
fp2.close()

for i in daf:
    url3 = 'http://yang.lzu.edu.cn/data/device_motion/anxiety/female/'+i.strip()[2:-2]+'_device_motion.json'
    r3 = requests.get(url3)
    fp3 = open(r'D:\data\device_motion\anxiety\female\{}_device_motion.json'.format(i.strip()[2:-2]),mode='w')
    fp3.write(r3.text)
fp3.close()

for i in dhf:
    url4 = 'http://yang.lzu.edu.cn/data/device_motion/health/female/'+i.strip()[2:-2]+'_device_motion.json'
    r4 = requests.get(url4)
    fp4 = open(r'D:\data\device_motion\health\female\{}_device_motion.json'.format(i.strip()[2:-2]),mode='w')
    fp4.write(r4.text)
fp4.close()

for i in gaf:
    url5 = 'http://yang.lzu.edu.cn/data/gyroscope/anxiety/female/'+i.strip()[2:-2]+'_gyroscope.json'
    r5 = requests.get(url5)
    fp5 = open(r'D:\data\gyroscope\anxiety\female\{}_gyroscope.json'.format(i.strip()[2:-2]),mode='w')
    fp5.write(r5.text)
fp5.close()

for i in ghf:
    url6 = 'http://yang.lzu.edu.cn/data/gyroscope/health/female/'+i.strip()[2:-2]+'_gyroscope.json'
    r6 = requests.get(url6)
    fp6 = open(r'D:\data\gyroscope\health\female\{}_gyroscope.json'.format(i.strip()[2:-2]),mode='w')
    fp6.write(r6.text)
fp6.close()

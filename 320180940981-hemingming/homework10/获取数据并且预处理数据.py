import requests
import pandas as pd
# get data from link
url = 'http://yang.lzu.edu.cn/data/index.txt'
response = requests.get(url)
response.encoding = 'utf-8'
data1 = response.text.replace('\n', ',').replace('./', '').split(',')

for i in range(len(data1)):
    if data1[i].endswith('json'):
        data_url = url[0:-9] + data1[i]
        if data1[i].split('/')[-1][-4:] == 'json':
            data = requests.get(data_url)
        data_url = url[0:-9]+data1[i]
        file = open('d:/python/file/homework10/数据集/' + data1[i].replace('/', '_'), 'wb')
        for j in data.iter_content(100000):
            file.write(j)
        file.close()

# The data is preprocessed
for fn in data1:                                        # preprocessing accelerometer
    if fn.startswith('acc') and fn.endswith('.json'):
        fname = fn.strip('./').replace('/', '_')
        df = pd.read_json("d:/python/file/homework10/数据集/"+fname)
        if df.empty:   # calculate the answer time
            time = 0
        else:
            time = df.iloc[:, 0].size // (5*60)
        if 8 < time < 60:   # reject unqualified data
            df = df.iloc[500:]
            var = df['x'].var()   # calculate the variance of the x-axis
            if var > 0.001:
                df.to_json("d:/python/file/homework10/预处理后数据/"+fname, orient='records')

for fn in data1:                                          # preprocessing devive_motion
    if fn.startswith('dev') and fn.endswith('.json'):
        fname = fn.strip('./').replace('/', '_')
        df = pd.read_json("d:/python/file/homework10/数据集/"+fname)
        if df.empty:            # calculate the answer time
            time = 0
        else:
            time = df.iloc[:, 0].size // (5*60)
        if 8 < time < 60:           # reject unqualified data
            df = df.iloc[500:]
            var = df['alpha'].var()
            if var > 3:
                df.to_json("d:/python/file/homework10/预处理后数据/"+fname, orient='records')

for fn in data1:                                          # proprecessing gyroscope
    if fn.startswith('gyr') and fn.endswith('.json'):
        fname = fn.strip('./').replace('/', '_')
        df = pd.read_json("d:/python/file/homework10/数据集/"+fname)
        if df.empty:                 # calculate the answer time
            time = 0
        else:
            time = df.iloc[:, 0].size // (5*60)
        if 8 < time < 60:                # reject unqualified data
            df = df.iloc[500:]
            df.to_json("d:/python/file/homework10/预处理后数据/"+fname, orient='records')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

times = []

with open('index.txt','r+') as f:
    for data in f:
        data = data.strip()
        if os.path.isfile(data):
            if os.path.getsize(data) < 5: #删除空文件
                os.remove(data)
                continue
            df = pd.read_json(data)
            time = int(len(df)/(5*60))
            times.append(time)

plt.figure(figsize=(8, 4), dpi=100)

bins=np.arange(0,150,10)
plt.rcParams['font.sans-serif']=['SimHei']
plt.title("时间分布图")
plt.xlabel("time")
plt.ylabel("number")
plt.hist(times,bins)
plt.savefig("时间分布图.jpg")

with open('index.txt','r+') as f:
    for data in f:
        data = data.strip()
        if os.path.isfile(data):
            df = pd.read_json(data)
            time = int(len(df)/(5*60))
            if time < 5 or time > 80:
                os.remove(data)
                

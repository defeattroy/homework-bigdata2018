import os
import pandas as pd


pathdir = os.listdir("E:\\qq\\data\\accelerometer_health_female1")
for s in pathdir:
    newdir=os.path.join("E:\\qq\\data\\accelerometer_health_female1",s)
    d=pd.read_json(newdir)
    time = d.iloc[:,0].size // (5*60)
    if 7 < time < 60:
        d = d.iloc[500:]
        var = d['x'].var()
        if var > 0.01:
            print(newdir)
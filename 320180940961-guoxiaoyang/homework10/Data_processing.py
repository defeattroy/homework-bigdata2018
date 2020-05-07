import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dp = os.listdir("E:\\python_src\\data\\gyroscope_anxiety_female")
for a in dp:
    ndp=os.path.join("E:\\python_src\\data\\gyroscope_anxiety_female",a)
    d=pd.read_json(ndp)
    d.plot()
    time = d.iloc[:,0].size // (5*40)
    if 5 < time < 40:
        d = d.iloc[500:]
        var = d['x'].var()
        if var > 0.01:
            print(ndp)

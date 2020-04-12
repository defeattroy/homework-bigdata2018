import pandas as pd
import matplotlib.pyplot as plt
import os

def cap(x,quantile=[0.01,0.99]):#quantile指范围，默认凡小于百分之1分位数和大于百分之99分位数的值将会被百分之1分位数和百分之99分位数替代
    Q01,Q99=x.quantile(quantile).values.tolist()
    if Q01>x.min():
        x=x.copy()
        x.loc[x<Q01]=Q01
    if Q99<x.max():
        x=x.copy()
        x.loc[x>Q99]=Q99
    return(x)
acc_anx = pd.read_csv('E:/jupyter-notebook/acc-hea-fm.csv')
new = acc_anx.apply(cap,quantile=[0.01,0.99])#转换
new.hist(bins=100)
plt.show()

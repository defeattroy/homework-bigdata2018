import pandas as pd
import matplotlib.pyplot as plt
import os

def cap(x,quantile=[0.01,0.99]):
    Q01,Q99=x.quantile(quantile).values.tolist()
    if Q01>x.min():
        x=x.copy()
        x.loc[x<Q01]=Q01
    if Q99<x.max():
        x=x.copy()
        x.loc[x>Q99]=Q99
    return(x)

acc_anx = pd.read_csv('E:/jupyter-notebook/dev-anx-fm.csv')
new = acc_anx.apply(cap,quantile=[0.01,0.99])
new.hist(bins=100)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os

HOUSING_PATH="E:/jupyter-notebook"
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"gyr-anx-fm.csv")
    return pd.read_csv(csv_path)

housing=load_housing_data()
#绘制直方图
housing.hist(bins=100,figsize=(15,10))#bins表示直方图中柱子的数量，figsize是每张图的大小
plt.show()

import doctest
import pandas as pd

#json文件转化为csv文件处理与accelerometer相同
gyr_anx = pd.read_csv('E:/jupyter-notebook/gyr-anx-fm.csv')
gyr_hea = pd.read_csv('E:/jupyter-notebook/gyr-hea-fm.csv')

'''
>>> gyr_anx.head()
          x         y         z
0 -0.403311  0.483386  0.043441
1 -0.325137  0.372052  0.051819
2 -0.352207  0.375036  0.037106
3 -0.159628  0.070110 -0.012776
4 -0.152926  0.099257  0.020089
>>> gyr_hea.head()
          z         x         y
0 -0.025757 -0.029282  0.119186
1 -0.051544 -0.280151  0.158722
2  0.031235  0.112396  0.024597
3  0.017914  0.099731 -0.012680
4 -0.011703  0.075333 -0.002457
>>> gyr_anx.shape
(114550, 3)
>>> gyr_hea.shape
(135253, 3)
>>> gyr_anx.isnull()
            x      y      z
0       False  False  False
1       False  False  False
2       False  False  False
3       False  False  False
4       False  False  False
...       ...    ...    ...
114545  False  False  False
114546  False  False  False
114547  False  False  False
114548  False  False  False
114549  False  False  False

[114550 rows x 3 columns]
>>> gyr_anx.isnull().any()
x    True
y    True
z    True
dtype: bool
>>> #存在缺失值
>>> missing=gyr_anx.columns[gyr_anx.isnull().any()].tolist()
>>> gyr_anx[missing].isnull().sum()
x    4
y    4
z    4
dtype: int64
>>> #缺失值相对于整体数据来说并不算多，我们可以知道是ID为./gyroscope/anxiety/female/20191114161025_2943_gyroscope.json
#./gyroscope/anxiety/female/20191114171226_3006_gyroscope.json、./gyroscope/anxiety/female/20191116154726_3651_gyroscope.json
#以及./gyroscope/anxiety/female/20191118173012_3937_gyroscope.json的设备数据缺失，可能因为不具备陀螺仪
>>> gyr_anx['x']=gyr_anx['x'].fillna(0)#将缺失值全部填充为0
gyr_anx['y']=gyr_anx['y'].fillna(0)
gyr_anx['z']=gyr_anx['z'].fillna(0)
SyntaxError: multiple statements found while compiling a single statement
>>> gyr_anx['x']=gyr_anx['x'].fillna(0)
>>> gyr_anx['y']=gyr_anx['y'].fillna(0)
>>> gyr_anx['z']=gyr_anx['z'].fillna(0)
>>> #将缺失值填充为0
>>> gyr_hea.isnull().any()#gyr_hea同样都存在缺失值
z    True
x    True
y    True
dtype: bool
>>> missing2 = gyr_hea.columns[gyr_hea.isnull().any()].tolist()
>>> gyr_hea[missing2].isnull().sum()
z    1
x    1
y    1
dtype: int64
>>> #缺失数据的ID为./gyroscope/health/female/20191111111559_1665_gyroscope.json
>>> gyr_hea['x']=gyr_hea['x'].fillna(0)#将缺失值全部填充为0
>>> gyr_hea['y']=gyr_hea['y'].fillna(0)
>>> gyr_hea['z']=gyr_hea['z'].fillna(0)
>>> 

'''
if __name__=='__main__':
  doctest.testmod(verbose=True)


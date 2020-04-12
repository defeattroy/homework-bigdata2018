import pandas as pd
import doctest

#json文件转化为csv文件处理与accelerometer相同
dev_anx = pd.read_csv('E:/jupyter-notebook/dev-anx-fm.csv')
dev_hea = pd.read_csv('E:/jupyter-notebook/dev-hea-fm.csv')

'''
>>> dev_anx.head()
       alpha       beta  gamma
0  246.90000 -39.649998   7.56
1  246.81999 -38.750000   6.75
2  246.73000 -37.110000   5.06
3  246.51000 -36.540000   4.66
4  247.34000 -36.270000   4.62
>>> dev_hea.head()
      gamma      alpha       beta
0  2.086156  239.30652 -32.816860
1  1.757870  231.14922 -33.373573
2  1.933872  229.72934 -34.644585
3  1.875400  229.94286 -35.494255
4  1.616465  230.43855 -35.866077
>>> dev_anx.shape
(475634, 3)
>>> dev_hea.shape
(452889, 3)
>>> dev_anx.isnull()
        alpha   beta  gamma
0       False  False  False
1       False  False  False
2       False  False  False
3       False  False  False
4       False  False  False
...       ...    ...    ...
475629  False  False  False
475630  False  False  False
475631  False  False  False
475632  False  False  False
475633  False  False  False

[475634 rows x 3 columns]
>>> dev_anx.isnull().any()
alpha    False
beta     False
gamma    False
dtype: bool
>>> dev_hea.isnull()
        gamma  alpha   beta
0       False  False  False
1       False  False  False
2       False  False  False
3       False  False  False
4       False  False  False
...       ...    ...    ...
452884  False  False  False
452885  False  False  False
452886  False  False  False
452887  False  False  False
452888  False  False  False

[452889 rows x 3 columns]
>>> dev_hea.isnull().any()
gamma    False
alpha    False
beta     False
dtype: bool
>>> #同样都没有缺失值

'''
if __name__=='__main__':
  doctest.testmod(verbose=True)

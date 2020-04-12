import doctest
import pandas as pd
#读取csv文件
acc_anx = pd.read_csv('E:/jupyter-notebook/acc-anx-fm.csv')
acc_hea = pd.read_csv('E:/jupyter-notebook/acc-hea-fm.csv')

'''
>>> acc_anx.head()
          x         y         z
0 -0.111657 -0.640157 -0.741568
1 -0.145579 -0.633443 -0.756502
2 -0.093783 -0.627508 -0.732474
3 -0.089605 -0.595811 -0.811558
4 -0.092811 -0.590913 -0.798304
>>> acc_hea.head()
          z         x         y
0 -0.806842 -0.062013 -0.573547
1 -0.842946 -0.042691 -0.556908
2 -0.850320 -0.020689 -0.530141
3 -0.807942 -0.030241 -0.566603
4 -0.811008 -0.032301 -0.576804
>>> acc_anx.shape
(275163, 3)
>>> acc_hea.shape
(328090, 3)
>>> #分析缺失数据，把对应的所有数据的位置列出来，元素为空则为True，否则为False
>>> acc_anx.isnull()
            x      y      z
0       False  False  False
1       False  False  False
2       False  False  False
3       False  False  False
4       False  False  False
...       ...    ...    ...
275158  False  False  False
275159  False  False  False
275160  False  False  False
275161  False  False  False
275162  False  False  False

[275163 rows x 3 columns]
>>> #列级别的判断，只要该列为空的元素，就为True，否则False
>>> acc_anx.isnull().any()
x    False
y    False
z    False
dtype: bool
>>> #acc_hea同理
>>> acc_hea.isnull()
            z      x      y
0       False  False  False
1       False  False  False
2       False  False  False
3       False  False  False
4       False  False  False
...       ...    ...    ...
328085  False  False  False
328086  False  False  False
328087  False  False  False
328088  False  False  False
328089  False  False  False

[328090 rows x 3 columns]
>>> acc_hea.isnull().any()
z    False
x    False
y    False
dtype: bool
>>> #可以看到，acc_hea以及acc_anx均没有缺失值

'''

if __name__=='__main__':
  doctest.testmod(verbose=True)

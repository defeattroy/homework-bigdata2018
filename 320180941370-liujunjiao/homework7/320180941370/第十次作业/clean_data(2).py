import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_data=pd.read_csv('抑郁症数据.csv',index_col=[0])
df_data['x']=pd.to_numeric(df_data['x'])
df_data['y']=pd.to_numeric(df_data['y'])
df_data['z']=pd.to_numeric(df_data['z'])

#去除手机浮动较小（可能将手机放置在桌面上）的数据
#可通过方差检验波动程度
fluctuate=df_data.groupby(['type','number']).var()
fluctuate=fluctuate.sort_index(level='type')
accelerometer=fluctuate.loc[('accelerometer',slice(None)),:]
accelerometer=accelerometer.reset_index('type')
accelerometer['x']=pd.to_numeric(accelerometer['x'])
accelerometer['y']=pd.to_numeric(accelerometer['y'])
accelerometer['z']=pd.to_numeric(accelerometer['z'])
sns.swarmplot(data=accelerometer)

#可发现当方差超过0.07时离散值较大，可能为陀螺仪损坏，可删除
special= accelerometer[(accelerometer['x']>0.07)].index.tolist()
special=special+accelerometer[(accelerometer['y']>0.07)].index.tolist()
special=special+accelerometer[(accelerometer['z']>0.07)].index.tolist()
#若方差太小，可能是因为平放在在桌面或是其他区域，可剔除
special=special+accelerometer[(accelerometer['x']<0.001)].index.tolist()
special=special+accelerometer[(accelerometer['y']<0.001)].index.tolist()
special=special+accelerometer[(accelerometer['z']<0.001)].index.tolist()
#剔除异常值
df_data=df_data[~df_data['number'].isin(special)]

#将清理完成的数据重新写入csv文件
df_data.to_excel('抑郁症数据（修改后）.xlsx')
df_data.to_csv('抑郁症数据（修改后）.csv')
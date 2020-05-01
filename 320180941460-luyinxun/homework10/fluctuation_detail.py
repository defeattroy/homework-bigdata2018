import pandas as pd
import matplotlib.pyplot as plot


#大幅波动较多
df0 = pd.read_json('d:\\数据集\\数据\\device_motion_anxiety_female_20191112201538_2281_device_motion.json')
high_level,low_level = 350,150  #define the Upper and lower thresholds
abnormal = []
for i in df0.alpha[:]:
    if i > high_level or i < low_level:
        abnormal.append(i)
df1=df0[-df0['alpha'].isin(abnormal)]  #the dataframe without wrong data
print(df1.plot())
d = pd.DataFrame(abnormal)
print(df1.plot)
print(d)



#大幅波动较少
df2 = pd.read_json('d:\\数据集\\数据\\device_motion_health_female_20191108092842_467_device_motion.json')
high = df2.alpha.quantile(q=0.75)
low = df2.alpha.quantile(q=0.25)

threshold = 100  #控制阈值
abnormal = []
for i,j in zip(df2.alpha[1:],df2.alpha[0:]):
    if(abs(i-j) > threshold) or (j > high + 1.5*(high - low)) or (j < low - 1.5*(high - low)):
        #print(j)
        abnormal.append(j)  #the list of abnormal data
df3=df2[-df2['alpha'].isin(abnormal)]
print(df3.plot())
d = pd.DataFrame(abnormal)
print(df2.plot)
print(d)

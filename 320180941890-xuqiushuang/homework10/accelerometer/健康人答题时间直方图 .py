#由于采集数据的频率是5HZ，我们可以得到每一个ID采集数据所需要的时间
#acc_hea：14.3min、14.0min、16.4min、14.6min、17.11min、35.8min、22.7min、18.3min、81.9min、21.8min、16.3min、13.7min、13.6min、9.6min、15.8min、15.1min、
#14.6min、98.5min、71.2min、16.2min、20.3min、12.2min、11.0min、10.4min、21.4min、14.9min、74.8min、87.6min、51.2min、14.9min、15.4min、9.8min、55.7min、
#12.9min、15.6min、49.8min、12.9min、15.5min、11.2min、14.4min、20.9min

# matplotlib模块绘制直方图
import matplotlib.pyplot as plt

# 读入数据
time=[14.3,14.0,16.4,14.6,17.11,35.8,22.7,18.3,81.9,21.8,16.3,13.7,13.6,9.6,15.8,15.1,14.6,98.5,71.2,16.2,
      20.3,12.2,11.0,10.4,21.4,14.9,74.8,87.6,51.2,14.9,15.4,9.8,55.7,12.9,15.6,49.8,12.9,15.5,11.2,14.4,
      20.9]
# 绘制直方图
plt.hist(x = time, # 指定绘图数据
         bins = 41, # 指定直方图中条块的个数
         color = 'steelblue', # 指定直方图的填充色
         edgecolor = 'black' # 指定直方图的边框色
         )
# 添加x轴和y轴标签
plt.xlabel('time')
plt.ylabel('frequency')
# 添加标题
plt.title('acc_hea')
# 显示图形
plt.show()

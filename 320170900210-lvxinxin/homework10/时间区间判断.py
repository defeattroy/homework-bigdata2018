import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties   
import pandas as pd
import os
file_path=r'D:\data\accelerometer'
xaxis=['time=0', '0<time<=10', '10<time<=20', '20<time<=30','30<time<=40', '40<time<=50',
       '50<time<=60', '60<time<=70', '70<time<=80', '80<time<=90','90<time<=100', 'time>100']
time_dic={}.fromkeys(xaxis, 0) 
def time_deal(file_path):
    for root,dirs,files in os.walk(file_path):
        for file in files:
            filename=os.path.join(root,file)
            #print(filename)
            if filename[-5:]==".json":
                with open(filename,'r',encoding='utf-8') as f:
                    data=json.load(f)
                    count_ac=len(data)
                    time_ac=count_ac/5/60
                    if time_ac==0:time_dic['time=0']+=1
                    elif 0<time_ac<10:time_dic[ '0<time<=10']+=1
                    elif 10<time_ac<20:time_dic[ '10<time<=20']+=1
                    elif 20<time_ac<30:time_dic[ '20<time<=30']+=1
                    elif 30<time_ac<40:time_dic[ '30<time<=40']+=1
                    elif 40<time_ac<50:time_dic[ '40<time<=50']+=1
                    elif 50<time_ac<60:time_dic[ '50<time<=60']+=1
                    elif 60<time_ac<70:time_dic[ '60<time<=70']+=1
                    elif 70<time_ac<80:time_dic[ '70<time<=80']+=1
                    elif 80<time_ac<90:time_dic[ '80<time<=90']+=1
                    elif 90<time_ac<100:time_dic[ '90<time<=100']+=1
                    else :time_dic[ 'time>100']+=1
       
    #print(time_dic)
    fig=plt.figure(figsize=(19,9))
    plt.bar(list(time_dic.keys()), list(time_dic.values()))
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
    plt.xlabel('所用时间（单位：分钟）', FontProperties=font)
    plt.ylabel('该区间人数', FontProperties=font)
    plt.title('问卷时间分布图', FontProperties=font)
    plt.xticks(rotation=20,fontsize=10) 
    plt.savefig('./问卷时间分布图.jpg')
    #时间10-90分钟合适

                    
if __name__=="__main__": 
    file_path=r'D:\data\accelerometer'
    time_deal(file_path)

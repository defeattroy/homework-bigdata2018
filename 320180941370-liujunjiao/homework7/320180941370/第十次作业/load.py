import urllib.parse
import time
import json
import pandas as pd
                                                    
def get_page(url):
    headers={
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.79 Safari/537.36'
}
    req = urllib.request.Request(url=url,headers=headers)
    res = urllib.request.urlopen(req)
    html = res.read().decode("utf-8")
    return html

def getDataList():
    dataList = []
    file_path ='D:\mypython\python程序\抑郁症数据.txt'
    with open(file_path,'r', encoding='utf-8') as file_object:
        for line in file_object:
            dataList.append(line.rstrip())
    return dataList

def geturl(dataList):
    url=[]
    for data in dataList:
        url.append('http://yang.lzu.edu.cn/data/{}'.format(data[2:]))
    return url

def get_data(ul,html,my_data):
    html = json.loads(html)
    c = len(html)
    for i in range(0,c):
        try:
            x=html[i]['x']
            y=html[i]['y']
            z=html[i]['z']
            ul = ul.replace('/','.')
            informationList=ul.split('.')
            my_data.append([informationList[7],informationList[8],informationList[10][0:14],x,y,z])
        except:
            alpha=html[i]['alpha']
            beta=html[i]["beta"]
            gamma=html[i]["gamma"]
            ul = ul.replace('/','.')
            informationList=ul.split('.')
            my_data.append([informationList[7],informationList[8],informationList[10][0:14],alpha,beta,gamma])
        print(my_data[-1])

my_data=[]
datalist=getDataList()
url=geturl(datalist)
for ul in url:
    get_data(ul,get_page(ul),my_data)

#将生成的列表转化为dataframe类型
df_data = pd.DataFrame(my_data)
df_data.columns = ['type','health','number','x','y','z']

df_data.head()
df_data.to_excel('抑郁症数据.xlsx')
df_data.to_csv('抑郁症数据.csv')
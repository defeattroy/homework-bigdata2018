import requests
import time

#下载数据

url = 'http://yang.lzu.edu.cn/data/'
file = requests.get(url + 'index.txt')  #get all the data-file's names from the main url
file.raise_for_status()
file_list = file.text.replace('\n',',').replace('./','').split(',')  #replace the symbol to recompose the path of detail data
      
for u in range(len(file_list)):
    data_url = url + file_list[u]    #get every data's url
    if file_list[u].split('/')[-1][-4:] == 'json':   #distinguish the data and index
        data = requests.get(data_url, timeout = 200)
        data.raise_for_status()
        
        file = open('d://数据集//数据//'+ file_list[u].replace('/','_'), 'wb')   #create every data-file's name   
        for i in data.iter_content(100000):
            file.write(i)
        file.close()
        print(file_list[u].split('/')[-1])

        time.sleep(0.5)

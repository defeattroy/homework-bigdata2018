#删除时间不在适宜范围的数据
import os,json
def del_time(file_path):
    for root,dirs,files in os.walk(file_path):
        for file in files:
            filename=os.path.join(root,file)
            #print(filename)
            if filename[-5:]==".json":
                with open(filename,'r',encoding='utf-8') as f:
                    data=json.load(f)
                    count_ac=len(data)
                    time_ac=count_ac/5/60
                if(time_ac<10 or time_ac>90):
                    os.remove(filename)
                    print('remove',filename)
    
if __name__=="__main__":
    file_path=r'D:\data\accelerometer'
    del_time(file_path)

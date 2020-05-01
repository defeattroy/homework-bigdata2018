'''针对所有文件删除空文件'''
import os
def get_path(file_path):
    for root,dirs,files in os.walk(file_path):
        for file in files:
            filename=os.path.join(root,file)
            del_file(filename)
def del_file(filename):
    size=os.path.getsize(filename)
    if size<1024:        #os.getsize获取大小单位为字节
        print("remove",filename)
        os.remove(filename)
if __name__=="__main__":
    file_path=r'D:\data'
    get_path(file_path)

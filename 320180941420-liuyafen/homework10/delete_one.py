import os

"""
删除所有没有存数据的小文件
函数的参数s是字符串型，代表文件路径
"""

def deleteFile_one(s):
    files = os.listdir(s)
    for file in files:
        b = s + "//" + file
        size = os.path.getsize(b)
        file_size = 2*1024
        if size <= file_size:
            os.remove(b)
            print(file + " deleted.")
    return


"""
调用函数删除6个文件夹里的小文件
"""

way1 = 'C:\\homework10\\accelerometer\\anxiety'
way2 = 'C:\\homework10\\accelerometer\\health'
way3 = 'C:\\homework10\\device_motion\\anxiety'
way4 = 'C:\\homework10\\device_motion\\health'
way5 = 'C:\\homework10\\gyroscope\\anxiety'
way6 = 'C:\\homework10\\\gyroscope\\health'

if __name__=='__main__':
    doctest.testmod(verbose=True)
    deleteFile_one(way1)
    deleteFile_one(way2)
    deleteFile_one(way3)
    deleteFile_one(way4)
    deleteFile_one(way5)
    deleteFile_one(way6)

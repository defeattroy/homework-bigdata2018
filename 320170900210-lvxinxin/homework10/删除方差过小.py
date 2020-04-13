def del_ac(fpath):
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            df = pd.read_json(path,encoding='utf-8')
            df = df.iloc[500:]
            var = df['x'].var()                    #计算x轴的方差
            if var <0.001:                        #剔除方差过小的无效数据
                os.remove(filename)
                print('remove',filename)
                
def del_de(file_path):
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            df = pd.read_json(path,encoding='utf-8')
            df = df.iloc[500:]
            var = df['alpha'].var()                    #计算alpha的方差
            if var <3:                        #剔除方差过小的无效数据
                os.remove(filename)
                print('remove',filename)
                


if __name__=="__main__":
    file_path=r'D:\data\accelerometer'
    del_ac(file_path)
    del_de(file_path)

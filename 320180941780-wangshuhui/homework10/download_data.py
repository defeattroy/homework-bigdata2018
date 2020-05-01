import requests
import os

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36 OPR/67.0.3575.115'}

def get_index(url):
    response = requests.get(url, headers=headers)
    with open('index.txt', 'w+') as f:
        f.write(response.text)
        f.close()

def get_data(url):
    with open('index.txt', 'r+') as f:
        flag = 0
        for data in f:
            data = data.strip()
            index = data.find('/')
            if data[-5:] != '.json':
                if flag != 0:
                    os.chdir('..')
                    os.chdir('..')
                    os.chdir('..')
                os.makedirs(data)
                os.chdir(data)
                flag=1
            else:
                link = url + data[index:]
                r = requests.get(link, headers=headers)
                name = os.path.basename(data)
                with open(name, 'w+') as d:
                    d.write(r.text)
                    d.close()

if __name__ == '__main__':
    url = 'http://yang.lzu.edu.cn/data'
    index_url = url + '/index.txt'
    get_index(index_url)
    get_data(url)

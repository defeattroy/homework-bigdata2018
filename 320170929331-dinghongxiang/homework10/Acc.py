# -*- coding: utf-8 -*-
"""
@Time £º 2020/4/12 19:27
@Auth £º Erris
@Version:  Python 3.8.0

"""

import os
import json
import urllib.request
import collections

Loc = 'http://yang.lzu.edu.cn/data/index.txt'
Loc_root = 'http://yang.lzu.edu.cn/data/'


# requests
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0', 'Connection': 'keep-alive' }
req = urllib.request.Request(url = Loc, headers = headers)
Locs = urllib.request.urlopen(req, timeout=60).read().decode('UTF-8').split('\n')

# get index.txt
Locs = [Loc_root + x.replace('./', '') for x in Locs if x.endswith('n')]


# check and create dir
if not os.path.exists('./DataSet/AccData/'):
    os.makedirs('./DataSet/AccData/')


# get files
for url in Locs:
    req = urllib.request.Request(url = url, headers = headers)
    cnt = urllib.request.urlopen(req, timeout=60).read()
    with open('./DataSet/AccData/' + url.replace('http://yang.lzu.edu.cn/data//', '').replace('http://yang.lzu.edu.cn/data/', '').replace('_accelerometer.json', '').replace('_device_motion.json', '').replace('_gyroscope.json', '').replace('/', '_') + '.txt', 'wb') as file:
        file.write(cnt)


file_dir = ['./DataSet/AccData/' + x for x in os.listdir('./DataSet/AccData')]


# delete empty files by using len()
for dir in file_dir:
    with open(dir, 'r') as file:
        content = json.load(file)
    if len(content) == 0:
        os.remove(dir)
        file_dir.remove(dir)
        continue


# calculate the percentage of the most frequent data in each axis
def most_per(data:list, k = 3):
    if len(data) > 0:
        c = collections.Counter(val_alpha)
        rs = 0
        for item in c.most_common(k):
            rs += item[1]
        return rs/len(data)
    else:
        return 0.0
            

# delete the file if one's data are too close to each other
for dir in file_dir:
    with open(dir, 'r') as file:
        content = json.load(file)
        val_alpha = []
        val_beta = []
        val_gamma = []
        for cnt in content:
            if dir.find('device_motion') == -1:
                val_alpha.append(cnt['x'])
                val_beta.append(cnt['y'])
                val_gamma.append(cnt['z'])
                
            else:
                val_alpha.append(cnt['alpha'])
                val_beta.append(cnt['beta'])
                val_gamma.append(cnt['gamma'])
        file.close()    # close the file mannually to avoid crash
        if most_per(val_alpha, 100) > 0.95 or most_per(val_alpha, 100) < 0.05:
            os.remove(dir)
            file_dir.remove(dir)
            continue
        if most_per(val_beta, 100) > 0.95 or most_per(val_beta, 100) < 0.05:
            os.remove(dir)
            file_dir.remove(dir)
            continue
        if most_per(val_gamma, 100) > 0.95 or most_per(val_gamma, 100) < 0.05:
            os.remove(dir)
            file_dir.remove(dir)
            continue
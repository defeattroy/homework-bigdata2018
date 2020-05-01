# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:42:13 2020

@author: lxz
"""

import os
import json   
import numpy
data_dir = './data/'
#clean = []

def get_car_json_info(data_dir):
    # get json file_names
    all_json=os.listdir(data_dir)
    
    for j_name in all_json:
        # open json file
        j=open(data_dir+j_name)
        
        # load info in json
        info=json.load(j)
        if len(info) < 3000 or len(info) > 18000:
            clean.append(j_name)  
        else:
            b = []        
            for i in info[:300]:
                x = i['alpha']
                y = i['beta']
                z = i['gamma']
                dis = numpy.cbrt(x**2 + y**2 + z**2)
                b.append(dis)
        if (max(b)-min(b))>2.5:

            clean.append(j_name)

    
#get_car_json_info(data_dir)
dl = clean.copy()
data_dir1 = 'C:/Users/lxz/Desktop/新建文件夹/'
submit = 'C:/Users/lxz/Desktop/result/'
def data(data_dir1):
    # get json file_names
    all_json=os.listdir(data_dir1)
    x = set(all_json).difference(set(dl))
    
    for j_name in x:
        # open json file
        j=open(data_dir1+j_name)
        info=json.load(j)
        with open(submit+j_name, 'w') as f:
            json.dump(info, f)

data(data_dir1)





















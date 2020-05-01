# -*- coding: utf-8 -*-
"""
@Time £º 2020/4/12 19:27
@Auth £º Erris
@Version:  Python 3.8.0

"""

import os
import json
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

file_dir = ['./DataSet/AccData/' + x for x in os.listdir('./DataSet/AccData')]


# try to find whether there are normal fractions in dataset, but it turns out not
for dir in file_dir:
    with open(dir, 'r') as file:
        content = json.load(file)
        if dir.find('device_motion') == -1:
            val_x = []
            val_y = []
            val_z = []
            for item in content:
                val_x.append(item['x'])
                val_y.append(item['y'])
                val_z.append(item['z'])
            vx = np.array(val_x)
            vy = np.array(val_y)
            vz = np.array(val_z)
            u_x = vx.mean()
            std_x = vx.std()
            _, px = stats.kstest(vx, 'norm', (u_x, std_x))
            u_y = vy.mean()
            std_y = vy.std()
            _, py = stats.kstest(vy, 'norm', (u_y, std_y))
            u_z = vz.mean()
            std_z = vz.std()
            _, pz = stats.kstest(vz, 'norm', (u_z, std_z))
            
            if px > 0.005 or py > 0.005 or pz > 0.005:
                print('Norm', dir)
        else:
            val_a = []
            val_b = []
            val_g = []
            for item in content:
                val_a.append(item['alpha'])
                val_b.append(item['beta'])
                val_g.append(item['gamma'])
            va = np.array(val_a)
            vb = np.array(val_b)
            vg = np.array(val_g)
            u_a = va.mean()
            std_a = va.std()
            _, pa = stats.kstest(va, 'norm', (u_a, std_a))
            u_b = vb.mean()
            std_b = vb.std()
            _, pb = stats.kstest(vb, 'norm', (u_b, std_b))
            u_g = vg.mean()
            std_g = vg.std()
            _, pg = stats.kstest(vg, 'norm', (u_g, std_g))
            
            if pa > 0.005 or pb > 0.005 or pg > 0.005:
                print('Norm', dir)


# check whether the data which last too long are useless, the result is the fraction is very valuable and that is why I didn't delete these data
for dir in file_dir:
    with open(dir, 'r') as file:
        content = json.load(file)
        val_alpha = []
        val_beta = []
        val_gama = []
        for cnt in content:
            if dir.find('device_motion') == -1:
                val_alpha.append(cnt['x'])
                val_beta.append(cnt['y'])
                val_gama.append(cnt['z'])
            else:
                val_alpha.append(cnt['alpha'])
                val_beta.append(cnt['beta'])
                val_gama.append(cnt['gamma'])
    
    if len(val_alpha) > 18000:
        plt.hist(val_alpha, bins = 200, color = 'r')
    else:
        plt.hist(val_alpha, bins = 200)
    plt.show()
    
    '''
    plt.hist(val_beta, bins = 100)
    plt.show()
    plt.hist(val_gama, bins = 100)
    plt.show()
    '''
    alpha = np.array(val_alpha)
    u = alpha.mean()
    std = alpha.std()
    D, p = stats.kstest(alpha, 'norm', (u, std))
    print(D, p)
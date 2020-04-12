#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_preprocessing.py
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/8 21:35    x-prince0     1.0      a test of data preprocessing
"""

# import pandas/matplotlib/os

import pandas as pd
import matplotlib.pyplot as plt
import os

path_ah = 'D:\\LZU\\No.4\\data_science_programming\\data\\accelerometer_health_female'
path_aa = 'D:\\LZU\\No.4\\data_science_programming\\data\\accelerometer_anxiety_female'
path_dh = 'D:\\LZU\\No.4\\data_science_programming\\data\\devicemotion_health_female'
path_da = 'D:\\LZU\\No.4\\data_science_programming\\data\\devicemotion_anxiety_female'
path_gh = 'D:\\LZU\\No.4\\data_science_programming\\data\\gyroscope_health_female'
path_ga = 'D:\\LZU\\No.4\\data_science_programming\\data\\gyroscope_anxiety_female'


def readfile(path):
    """
    Reading the '.json' files and merging the data,
    and delete the empty files at the same time.
    :param path: the path of different categories of data
    :return: a list of data after merged
    """
    df_list = []
    file_list = os.listdir(path)
    fl_change = []

    for file in file_list:
        data = pd.read_json('{0}\\{1}'.format(path, file))  # 返回一个dataframe
        if not data.empty:
            df_list.append(data)
            fl_change.append(file)

    file_list = fl_change

    return df_list, file_list


# 将要处理的数据和文件路径生成列表
dl_ah, fl_ah = readfile(path_ah)
dl_aa, fl_aa = readfile(path_aa)
dl_dh, fl_dh = readfile(path_dh)
dl_da, fl_da = readfile(path_da)
dl_gh, fl_gh = readfile(path_gh)
dl_ga, fl_ga = readfile(path_ga)


def finish_time(df_list):
    """
    getting how much time they finish the test, then make a bar chart
    :param df_list: a dataframe list including the initial data
    :return: chart of time with this type of data (minute)
    """
    times = []

    for df in df_list:
        time = len(df) / 5 / 60
        #将时间控制在10~100min内
        if 10 < time < 100:
            times.append(time)

    times.sort()
    plt.bar(range(len(times)), times, alpha=0.5)


# accelerometer
finish_time(dl_ah)
finish_time(dl_aa)
# device motion
finish_time(dl_dh)
finish_time(dl_da)
# gyroscope
finish_time(dl_gh)
finish_time(dl_ga)

# 合并数据
data_ah = pd.concat(dl_ah)
data_aa = pd.concat(dl_aa)
data_dh = pd.concat(dl_dh)
data_da = pd.concat(dl_da)
data_gh = pd.concat(dl_gh)
data_ga = pd.concat(dl_ga)


def mean_std(df_list):
    """
    getting their means and standard deviation of these files
    :param df_list: a data frame list
    :return: mean and standard deviation of these data frames
    """
    df_means = []  # 平均值
    df_stds = []  # 标准差
    i = 1
    for df in df_list:
        mean = df.describe().iloc[1].to_frame().T
        std = df.describe().iloc[2].to_frame().T
        df_means.append(mean)
        df_stds.append(std)

    df_means.sort(key=df_means)
    df_stds.sort(key='x')
    return df_means, df_stds


# 赋值
ah_means, ah_stds = mean_std(dl_ah)
aa_means, aa_stds = mean_std(dl_aa)
dh_means, dh_stds = mean_std(dl_dh)
da_means, da_stds = mean_std(dl_da)
gh_means, gh_stds = mean_std(dl_gh)
ga_means, ga_stds = mean_std(dl_ga)


# 画出平均值的折线图
pd.concat(ah_means).plot()
plt.title('means of \'accelerometer\' and \'health\'')
pd.concat(aa_means).plot()
plt.title('means of \'accelerometer\' and \'anxiety\'')

pd.concat(dh_means).plot()
plt.title('means of \'device motion\' and \'health\'')
pd.concat(da_means).plot()
plt.title('means of \'device motion\' and \'anxiety\'')

pd.concat(gh_means).plot()
plt.title('means of \'gyroscope\' and \'health\'')
pd.concat(ga_means).plot()
plt.title('means of \'gyroscope\' and \'anxiety\'')


# 画出标准差的折线图
pd.concat(ah_stds).plot()
plt.title('standard deviations of \'accelerometer\' and \'health\'')
pd.concat(aa_stds).plot()
plt.title('standard deviations of \'accelerometer\' and \'anxiety\'')

pd.concat(dh_stds).plot()
plt.title('standard deviations of \'device motion\' and \'health\'')
pd.concat(da_stds).plot()
plt.title('standard deviations of \'device motion\' and \'anxiety\'')

pd.concat(gh_stds).plot()
plt.title('standard deviations of \'gyroscope\' and \'health\'')
pd.concat(ga_stds).plot()
plt.title('standard deviations of \'gyroscope\' and \'anxiety\'')
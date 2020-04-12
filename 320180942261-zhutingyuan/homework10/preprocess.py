#!/usr/bin/env python3
# Copyright (c) 2020-4 ZhuTingYuan(Refection). All rights reserved
# This is a preprocessing code for the given data
# You can run directly or import
# In this code, all the code for the drawing is commented, you can use them
# in jupyter notebook
# Directly run the file organization directory as:
# D:.
# │  preprocess.py
# │
# │
# └─data
#       20191107150410_145_accelerometer.json
#       20191107150410_145_device_motion.json
#       20191107150410_145_gyroscope.json
#       ......


# from matplotlib import pyplot as plt
import os
import pandas as pd
import math


def read_file(f):
    """
    Used to read a data file from a path, return string
    """
    f = "./data/" + f
    with open(f, 'r') as f:
        file_txt = f.read()
    return file_txt


def read_file_pd(f):
    """
    Used to read a data file from a path, return dataframe
    """
    f = "./data/" + f
    return pd.read_json(f)


def remove_file(f):
    """
    remove an key file in path
    """
    f = "./data/" + f
    try:
        os.remove(f)
        print("remove file :", f)
    except(Exception):
        print(f)
        print('File deleted!')


def all_files_remove_file(all_files):
    """
    remove all key files in path
    """
    for f in all_files:
        filename = f
        f = "./data/" + f
        with open(f, 'r') as fs:
            if(len(fs.read()) <= 2000):
                remove_file(filename)


def variance_nine(x_list):
    """
    Returns the variance of the list (default mean is 1g(9.8N/m**2))
    """
    m = len(x_list)
    return 1 / (m - 1) * sum((x + 1.0)**2 for x in x_list)


def z_cull(f, alpha_var=0.003):
    """
    Eliminate inappropriate files (possibly without a handheld device) by
    calculating the z-axis change
    """
    f_remove = []
    for f in f_accelerometer:
        file_json = read_file_pd(f)
        var_z = variance_nine(file_json['z'])
        if var_z <= 0.003:  # 0.003  == 1383,2535
            print("Calculate that the device may be placed on the table:{}".
                  format(f))
            f_remove.append(f)
            # file_json.plot(color=['#7FFFAA', '#6495ED', '#5F9EA0'], title=f)
    for f in f_remove:
        remove_file(f)


def mean(x_list):
    """return mean of the data"""
    return (sum(x for x in x_list) / len(x_list))


def variance(x_list):
    """return variance of the data"""
    m = len(x_list)
    return 1 / (m - 1) * sum((x - mean(x_list))**2 for x in x_list)


def standard_deviation(f):
    """
    return standard_deviation of the data
    """
    return math.sqrt(variance(f))


def remove_not_sitdown(f_pd):
    """
    remove the dataframe after deleting error data
    """
    test = ~f_pd.index.isin(range(0, int(len(f_pd) * 0.1)))
    f_pd = f_pd[test]
    f_pd.index = range(len(f_pd))
    return f_pd


def not_sitdown(f_accelerometer):
    """
    Calculate the standard deviation dev_1 of the first 10% and dev_2 of the
    last 90% respectively.
    Is that there is an unsettled phenomenon, the top 10% of the data was
    eliminated.
    """
    for f in f_accelerometer:
        f_pd = read_file_pd(f)
        x_dev_1 = standard_deviation(f_pd['x'][:int(len(f_pd) * 0.1)])
        y_dev_1 = standard_deviation(f_pd['y'][:int(len(f_pd) * 0.1)])
        z_dev_1 = standard_deviation(f_pd['z'][:int(len(f_pd) * 0.1)])
        dev_1 = (x_dev_1 + y_dev_1 + z_dev_1) / 3
        x_dev_2 = standard_deviation(f_pd['x'][int(len(f_pd) * 0.1):])
        y_dev_2 = standard_deviation(f_pd['y'][:int(len(f_pd) * 0.1):])
        z_dev_2 = standard_deviation(f_pd['z'][:int(len(f_pd) * 0.1):])
        dev_2 = (x_dev_2 + y_dev_2 + z_dev_2) / 3
        alpha = dev_1 / dev_2
        if alpha > 1.0:
            print('α: {}, file name is: {}'.format(alpha, f))
            # f_pd.plot(color=['#7FFFAA', '#6495ED', '#5F9EA0'], title=f)
            f_pd_after = remove_not_sitdown(f_pd)
            # f_pd_after.plot(color=['#7FFFAA', '#6495ED', '#5F9EA0'], title=f)
            f = "./data/" + f
            f_pd_after.to_json(f, orient='records')
            print("length of file before change: {}".format(len(f_pd)))
            print("length of file after change: {}".format(len(f_pd_after)))
            # plt.show()


def culling_function(f_pd):
    """
    determine the error estimate caused by the device crossing the baseline
    axis
    Returns a list of abnormal data
    """
    first = list(f_pd['alpha'])
    second = list(f_pd['alpha'])
    first.append(0)
    second.insert(0, 0)
    exception_list = []
    flag = 0
    for i, j in zip(first, second):
        if(80 > int(i) > -80 and 300 < int(j) < 360) or (80 > int(j) > -80 and
                                                         300 < int(i) < 360):
            exception_list.append(flag - 1)
            exception_list.append(flag)
        flag += 1
    return exception_list


def flip_data_culling(f, alpha=20):
    """
    For the error estimation caused by the equipment crossing the reference
    axis, establish a copy after removing the abnormal
    value([filename_copy].json)
    """
    f_pd = read_file_pd(f)
    except_list = []
    f_copy = "./data/" + f.split(".")[0] + '_copy.json'
    for i in range(0, alpha):
        except_list = culling_function(f_pd)
        test = ~f_pd.index.isin(except_list)
        f_pd = f_pd[test]
        f_pd.index = range(len(f_pd))
        except_list.clear()
    print("create copy file: {}".format(f_copy))
    f_pd.to_json(f_copy, orient='records')


def all_file_filp_data_culling(f_device_motion):
    '''
    To all files :for the error estimation caused by the equipment crossing
    the reference axis,establish a copy after removing the abnormal
    value([filename_copy].json)
    '''
    for f in f_device_motion:
        flip_data_culling(f, alpha=100)


if __name__ == "__main__":
    # path = os.getcwd()
    all_files = [f for f in os.listdir("./data/")]

    # Then, a list of data measured by different sensors is established:
    f_accelerometer = [x for x in all_files if x.find("accelerometer") + 1]
    f_gyroscope = [x for x in all_files if x.find("gyroscope") + 1]
    f_device_motion = [x for x in all_files if x.find("device_motion") + 1]

    # Get the length of different lists:
    print('len of f_accelerometer: {}'.format(len(f_accelerometer)))
    print('len of f_gyroscope: {}'.format(len(f_gyroscope)))
    print('len of f_device_motion: {}'.format(len(f_device_motion)))

    """
        First, the subjects of the same three groups of sensors were read in
    three data and observed
    """
    f1 = pd.read_json("./data/20191107150410_145_accelerometer.json")
    f2 = pd.read_json("./data/20191107150410_145_device_motion.json")
    f3 = pd.read_json("./data/20191107150410_145_gyroscope.json")
    """
        It can be found that, even for the same subject, the sample points
    obtained from sampling are different. Therefore, considering the loss of
    data, even if a subject has data files of three sensors, it is impossible
    to draw a conclusion simply by comparing the data of three sensors
    according to the time axis.
    """
    print(len(f1))
    print(len(f2))
    print(len(f3))

    # f1.plot(color=['#7FFFAA','#6495ED','#5F9EA0'])
    # f2.plot(color=['#7FFFAA','#6495ED','#5F9EA0'])
    # f3.plot(color=['#7FFFAA','#6495ED','#5F9EA0'])
    # plt.show()

    """
    1. The phone has no gyroscope, resulting in missing values.
    Action: delete the file.
        Considering that some of the tested equipment has no gyroscope,
    resulting in the data file being empty, through reading the length of the
    file, deleting the empty file, and considering the time required to answer
    the questionnaire, it is considered that the data obtained within 5 minutes
    is arbitrary for the experiment, and the data may not have reference value,
    Therefore, the files within 5 minutes (about 2000 data points) will also
    be eliminated when the empty files are eliminated.
        For the acceleration sensor, the z-axis component should be the
    gravitational acceleration, i.e. 9.8N / s * * 2, from which the
    acceleration unit is estimated as G
        For F_accelerometer, take its z component and investigate its change in
    time series. It is predicted that:Take the reference value
    1g (9.8N / m * * 2), and calculate the variance
    """
    all_files_remove_file(all_files)

    """
    2. The phone is lying flat on the table, causing data failure (as can be
    seen from the acceleration sensor z-axis).
    Action: delete the file
        For the acceleration sensor, the expected unit is g. According to the
    plot rvation data, the x,y and z line graphs of some data all have very
    small fluctuations, while the line graphs of z are stable around
    1g(9.8n /m**2), with almost no fluctuations (you can see images in pdf).
    Therefore, it is speculated that the mobile phone is placed flat on a plane
    (such as a table), and the data obtained does not meet the collection
    conditions, so it is necessary to delete such files:
        For the acceleration sensor, the z-axis component should be the
    gravitational acceleration, i.e. 9.8N/s**2, from which the acceleration
    unit is estimated as G
        For F_accelerometer, take its z component and investigate its change in
    time series. It is predicted that:
    Take the reference value 1g (9.8N / m * * 2), and calculate the variance
    """
    all_files = [f for f in os.listdir("./data/")]
    f_accelerometer = [x for x in all_files if x.find("accelerometer") + 1]
    f_gyroscope = [x for x in all_files if x.find("gyroscope") + 1]
    f_device_motion = [x for x in all_files if x.find("device_motion") + 1]
    z_cull(f_accelerometer)

    """
    3. Considering the unsettled data at the beginning, the unsettled data
    should be eliminated
    Action:eliminate the data that is considered to have problems.
    For acceleration sensor data:
        The time is time = (len / 5) / 60 minutes. Considering that the
    accelerometer is not set at the beginning, take the top 10% for the
    accelerometer, The first 10% and the last 90% of the standard deviation,
    respectively, are calculated, if dev ﹣ 1 / dev ﹣ 2 > α (α > 1)
    It is considered that there is an unsettled phenomenon, and the first 10%
    of the data are eliminated.
    """
    all_files = [f for f in os.listdir("./data/")]
    f_accelerometer = [x for x in all_files if x.find("accelerometer") + 1]
    f_gyroscope = [x for x in all_files if x.find("gyroscope") + 1]
    f_device_motion = [x for x in all_files if x.find("device_motion") + 1]
    not_sitdown(f_accelerometer)

    """
    4. Remove misreading caused by equipment crossing the reference axis
    (device_motion).
    Action: Remove the misread data and create the deleted copy(filename_copy)
        for device_motion:For alpha coordinates, it can be observed that there
    is a jump from 0-360, but this jump is actually caused by the equipment
    crossing the reference axis, resulting in the transfer of recorded data
    from 0-360 degree record to 360 degree record, which is not caused by
    violent movement. If the change range is considered, it will form a
    miscalculation, so this kind of data is recursively eliminated, However,
    the original data can obtain accurate real-time angle information,
    which may also be useful. Therefore, it is not directly removed from the
    original book data files, but for each device_motion file, a copy
    (device_motion_copy) after recursive elimination is established. When it is
    necessary to analyze the change of action, the data of device_motion_copy
    files is used. When it is necessary to accurately obtain real-time angle
    information, the original file (device_motion) is used.
    """
    # path = os.getcwd()
    all_file_filp_data_culling(f_device_motion)
    all_files = [f for f in os.listdir("./data/")]
    f_device_motion_copy = [
        x for x in all_files if x.find("device_motion_copy") + 1]

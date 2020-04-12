import os
import pandas as pd


path1 = 'E:/xldownload/data/accelerometer_anxiety_femal'
path2 = 'E:/xldownload/data/accelerometer_health_femal'
path3 = 'E:/xldownload/data/device_motion_anxiety_femal'
path4 = 'E:/xldownload/data/device_motion_health_femal'
path5 = 'E:/xldownload/data/gyroscope_anxiety_femal'
path6 = 'E:/xldownload/data/gyroscope_health_femal'
aa_timelist = []  # storage accelerometer_anxiety_femal's times
ah_timelist = []  # storage accelerometer_health_femal's times
dma_timelist = []  # storage device_motion_anxiety_femal's times
dmh_timelist = []  # storage device_motion_health_femal's times
ga_timelist = []  # storage gyroscope_anxiety_femal's times
gh_timelist = []  # storage gyroscope_health_femal's times
# get all filename in each folder
path1_list = os.listdir(path1)
path2_list = os.listdir(path2)
path3_list = os.listdir(path3)
path4_list = os.listdir(path4)
path5_list = os.listdir(path5)
path6_list = os.listdir(path6)

for aa_filename in path1_list:
    aa_name = (os.path.join(path1, aa_filename)).replace('\\', '/')
    df_aa2 = pd.read_json(aa_name)
    if df_aa2.isnull() is True:
        df_aa2.loc[df_aa2.isnull] = '0'  # replace missing values with 0
        time = ((len(df_aa2) / 5) / 60)
        aa_time = round(time, 2)  # keep 2 bit after point
        if 5 <= aa_time <= 50:
            aa_timelist.append(aa_time)
        else:
            os.remove(aa_name)  # delete the useless file
    else:
        time = ((len(df_aa2) / 5) / 60)
        aa_time = round(time, 2)
        if 5 <= aa_time <= 50:
            aa_timelist.append(aa_time)
        else:
            os.remove(aa_name)

for ah_filename in path2_list:
    ah_name = (os.path.join(path2, ah_filename)).replace('\\', '/')
    df_ah2 = pd.read_json(ah_name)
    if df_ah2.isnull() is True:
        df_ah2.loc[df_ah2.isnull] = '0'
        time = ((len(df_ah2) / 5) / 60)
        ah_time = round(time, 2)
        if 5 <= ah_time <= 50:
            ah_timelist.append(ah_time)
        else:
            os.remove(ah_name)
    else:
        time = ((len(df_ah2) / 5) / 60)
        ah_time = round(time, 2)
        if 5 <= ah_time <= 50:
            ah_timelist.append(ah_time)
        else:
            os.remove(ah_name)

for dma_filename in path3_list:
    dma_name = (os.path.join(path3, dma_filename)).replace('\\', '/')
    df_dma2 = pd.read_json(dma_name)
    if df_dma2.isnull() is True:
        df_dma2.loc[df_dma2.isnull] = '0'
        time = ((len(df_dma2) / 5) / 60)
        dma_time = round(time, 2)
        if 5 <= dma_time <= 50:
            dma_timelist.append(dma_time)
        else:
            os.remove(dma_name)
    else:
        time = ((len(df_dma2) / 5) / 60)
        dma_time = round(time, 2)
        if 5 <= dma_time <= 50:
            dma_timelist.append(dma_time)
        else:
            os.remove(dma_name)

for dmh_filename in path4_list:
    dmh_name = (os.path.join(path4, dmh_filename)).replace('\\', '/')
    df_dmh2 = pd.read_json(dmh_name)
    if df_dmh2.isnull() is True:
        df_dmh2.loc[df_dmh2.isnull] = '0'
        time = ((len(df_dmh2) / 5) / 60)
        dmh_time = round(time, 2)
        if 5 <= dmh_time <= 50:
            dmh_timelist.append(dmh_time)
        else:
            os.remove(dmh_name)
    else:
        time = ((len(df_dmh2) / 5) / 60)
        dmh_time = round(time, 2)
        if 5 <= dmh_time <= 50:
            dmh_timelist.append(dmh_time)
        else:
            os.remove(dmh_name)

for ga_filename in path5_list:
    ga_name = (os.path.join(path5, ga_filename)).replace('\\', '/')
    df_ga2 = pd.read_json(ga_name)
    if df_ga2.isnull() is True:
        df_ga2.loc[df_ga2.isnull] = '0'
        time = ((len(df_ga2) / 5) / 60)
        ga_time = round(time, 2)
        if 5 <= ga_time <= 50:
            ga_timelist.append(ga_time)
        else:
            os.remove(ga_name)
    else:
        time = ((len(df_ga2) / 5) / 60)
        ga_time = round(time, 2)
        if 5 <= ga_time <= 50:
            ga_timelist.append(ga_time)
        else:
            os.remove(ga_name)

for gh_filename in path6_list:
    gh_name = (os.path.join(path6, gh_filename)).replace('\\', '/')
    df_gh2 = pd.read_json(gh_name)
    if df_gh2.isnull() is True:
        df_gh2.loc[df_gh2.isnull] = '0'
        time = ((len(df_gh2) / 5) / 60)
        gh_time = round(time, 2)
        if 5 <= gh_time <= 50:
            gh_timelist.append(gh_time)
        else:
            os.remove(gh_name)
    else:
        time = ((len(df_gh2) / 5) / 60)
        gh_time = round(time, 2)
        if 5 <= gh_time <= 50:
            gh_timelist.append(gh_time)
        else:
            os.remove(gh_name)

print("accelerometer_anxiety times")
print(aa_timelist)
print("accelerometer_health times")
print(ah_timelist)
print("device_motion_anxiety times")
print(dma_timelist)
print("device_motion_health times")
print(dmh_timelist)
print("gyroscope_anxiety times")
print(ga_timelist)
print("gyroscope_health times")
print(gh_timelist)
import pandas as pd


df_aa1 = pd.read_json(r'20191119174618_4121_accelerometer.json')  # accelerometer_anxiety
df_ah1 = pd.read_json(r'20191120112519_4218_accelerometer.json')  # accelerometer_health
df_dma1 = pd.read_json(r'20191119174618_4121_device_motion.json')  # device_motion_anxiety
df_dmh1 = pd.read_json(r'20191120112519_4218_device_motion.json')  # device_motion_health
df_ga1 = pd.read_json(r'20191119174618_4121_gyroscope.json')  # gyroscope_anxiety
df_gh1 = pd.read_json(r'20191120112519_4218_gyroscope.json')  # gyroscope_health

a_namelist = [df_ga1, df_dma1, df_aa1]  # Anxiety people's data list
h_namelist = [df_gh1, df_dmh1, df_ah1]  # Health people's data list
namelist = [df_aa1, df_ah1, df_dma1, df_dmh1, df_ga1, df_gh1]  # All data list
a_lenlist = []  # Anxiety people's data length list
h_lenlist = []  # Health people's data length list

# Check missing values
for name in namelist:
    print(name.info())
    print("before processing:", name.isnull().sum())
    if name.isnull() is True:
        name.loc[name.isnull] = '0'  # replace missing values with 0
        print("after processing:", name.isnull().sum())

# Get time
for a_name in a_namelist:
    a_lenlist.append(len(a_name))
a_time = (max(a_lenlist) / 5) / 60
print('The time of anxiety people', a_time, 'min')

for h_name in h_namelist:
    h_lenlist.append(len(h_name))
h_time = (max(h_lenlist) / 5) / 60
print('The time of health people', h_time, 'min')
#将同一类型的数据汇总到一个json文件里，一共有六个类型，分别是acc-anx-fm.json、acc-hea-fm.json、
#dev-anx-fm.json、dev-hea-fm.json、gyr-anx-fm.json、gyr-hea-fm.json
import csv  #将json文件转化为csv文件
import simplejson as json

# 定义转换函数
def trans(json_path, csv_path):
    json_file = open(json_path, 'r', encoding='utf8') 
    csv_file = open(csv_path, 'a', newline='', encoding='utf8')
    writer = csv.writer(csv_file)
    # 读取json数据
    dic_data = json.load(json_file)

    keys = []
    for dic in dic_data:
        keys = dic.keys()
        # 写入列名
        writer.writerow(keys)
        break

    for dic in dic_data:
        for key in keys:
            if key not in dic:
                dic[key] = ''
        writer.writerow(dic.values())

    json_file.close()
    csv_file.close()

# 调用转换函数
json_path = 'E:/jupyter-notebook/accelerometer-anxiety-female/acc-anx-fm.json'
#或者accelerometer-health-female/acc-hea-fm.json
csv_path = 'E:/jupyter-notebook/acc-anx-fm.csv'
trans(json_path, csv_path)

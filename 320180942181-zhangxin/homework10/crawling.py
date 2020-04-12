import requests
import os


def crawl() -> str:
    """
    Crawl data we need from web page and save them in different directories on disk.
    :return: String of succeed information
    """
    # Create directories to save data
    try:
        os.makedirs('data/accelerometer/health')
        os.makedirs('data/accelerometer/anxiety')
        os.makedirs('data/device_motion/health')
        os.makedirs('data/device_motion/anxiety')
        os.makedirs('data/gyroscope/health')
        os.makedirs('data/gyroscope/anxiety')
    except Exception as err:
        print(err)
    # Index url
    url = 'http://yang.lzu.edu.cn/data/index.txt'
    # Head of the full file url
    part_url = 'http://yang.lzu.edu.cn/data/'
    # Http request
    response = requests.get(url)
    # Get the content of the index_txt
    index_txt = response.text
    pre_path_list = index_txt.split('./')
    # The list to save all paths
    path_list = []
    # print(index_txt)
    # print(pre_path_list)
    for i in pre_path_list:
        i = i.strip()  # Delete the space in the head or tail of the string
        i = i.replace('\n', '')
        if i.count('json') == 2:  # Handel path of special format
            special_url = i.split('json/')
            special_url[0] = special_url[0] + 'json'
            path_list.extend(special_url)
            continue
        if i.endswith('.json'):  # The path which has .json is the path we want
            path_list.append(i)
    # for i in path_list:
    #     print(i)
    # print(len(path_list))
    # Confirm the number of the paths is equal to the index_txt
    assert index_txt.count('json') == len(path_list)

    for path in path_list:
        full_url = part_url + path  # Get full url
        try:
            # Get the content of the data
            data_response = requests.get(full_url)
            data_json = data_response.text

            # data_name = 'data/' + '-'.join(path.split('/'))
            # Get the name of data to save them
            data_name = path.split('/')[-1:][0]
            # According to the classes of the data to choose path to save it
            if path.startswith('accelerometer'):
                path = path.replace('accelerometer/', '')
                if path.startswith('health'):
                    data_name = 'data/accelerometer/health/' + data_name
                else:
                    data_name = 'data/accelerometer/anxiety/' + data_name
            elif path.startswith('device_motion'):
                path = path.replace('device_motion/', '')
                if path.startswith('health'):
                    data_name = 'data/device_motion/health/' + data_name
                else:
                    data_name = 'data/device_motion/anxiety/' + data_name
            elif path.startswith('gyroscope'):
                path = path.replace('gyroscope/', '')
                if path.startswith('health'):
                    data_name = 'data/gyroscope/health/' + data_name
                else:
                    data_name = 'data/gyroscope/anxiety/' + data_name
            # Create a file to save data
            with open(data_name, mode='a', encoding='utf-8') as f_data:
                f_data.write(data_json)
            print(full_url)
            # Another way to save data
            # f_data = open(data_name, 'w')
            # f_data.write(data_json)
            # f_data.close()
        except Exception as err:
            print(err)
    return 'Crawling over!'


if __name__ == "__main__":
    crawl()

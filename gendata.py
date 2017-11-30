# -*- coding: utf-8 -*-
"""
将输入数据处理成每个mall的文件，添加wifi连接数量的筛选
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import gc

path = './'
spath = './data/'
df = pd.read_csv(path + u'训练数据-ccf_first_round_user_shop_behavior.csv')
shop = pd.read_csv(path + u'训练数据-ccf_first_round_shop_info.csv')
test = pd.read_csv(path + u'AB榜测试集-evaluation_public.csv')
df = pd.merge(df, shop[['shop_id', 'mall_id']], how='left', on='shop_id')
df['time_stamp'] = pd.to_datetime(df['time_stamp'])
train = pd.concat([df, test])
mall_list = list(set(list(shop.mall_id)))
result = pd.DataFrame()
count = 0

for mall in mall_list:
    traino1 = ''
    traino2 = ''
    train1 = train[train.mall_id == mall].reset_index(drop=True)
    l = []
    wifi_dict = {}
    for index, row in train1.iterrows():
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            row[i[0]] = int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]] = 1
            else:
                wifi_dict[i[0]] += 1
        l.append(row)
    delate_wifi = []
    for i in wifi_dict:
        if wifi_dict[i] < 20:
            delate_wifi.append(i)
    m = []
    m2 = []
    for row in l:
        new = {}
        new2 = {}
        for n in row.keys():
            new2[n] = row[n]
            if n not in delate_wifi:
                new[n] = row[n]
        m.append(new)
        m2.append(new2)
    traino1 = pd.DataFrame(m)

    count = count + 1
    print(traino1.shape)
    print(mall, '序号', count)

    with open((spath + mall + '_1.txt'), 'wb') as f:
        pickle.dump(traino1, f)
    # del traino1
    # del m
    # del m2
    # del l
    # del wifi_dict
    # gc.collect()
print('finish')

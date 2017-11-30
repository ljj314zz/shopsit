# -*- coding: utf-8 -*-
"""
输入每个mall文件，添加属性

"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
import pickle
import gc
import matplotlib.pyplot as plt
import xgboost as xgb
import itertools
import os
import math
import csv
import scipy as sp
import scipy.stats
from math import radians, cos, sin, asin, sqrt, pi, log, tan
from sklearn.cluster import KMeans
# import procewifi
from sklearn.cross_validation import train_test_split

path = '../'
from newmodel import sitechange


def geodistance(lng1, lat1, lng2, lat2):
    '''计算经纬度两点间距离-m'''
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * 6371 * 1000
    return dis


def millerxy(lon, lat):
    '''经纬度坐标转换平面坐标'''
    # pi=math.pi
    l = 6381372 * pi * 2
    w = l
    h = l / 2
    mill = 2.3
    x = lon * pi / 180
    y = lat * pi / 180
    y = 1.25 * math.log(math.tan(0.25 * pi + 0.4 * y))
    x = (w / 2) + (w / (2 * pi)) * x
    y = (h / 2) - (h / (2 * mill)) * y
    return x, y


def addfea(train1):
    '''增加平面坐标，星期，每天具体时间属性'''
    addfeature = []
    flagtwice = 0  # 顾客多次访问一个店铺
    tshape = train1.shape
    flagtime = []  #
    for ix, row in train1.iterrows():
        # 日期时间
        try:
            l2tmp = row['time_stamp']._time_repr.split(':')
        except:
            print(row['time_stamp'])
            l2tmp = row['time_stamp']._time_repr.split(':')
        tmplday = int(int(l2tmp[0]) * 6 + int(l2tmp[1]) / 10)
        if tmplday < 100:
            lday = True
        else:
            lday = False
        # lday = int(l2tmp[0])
        lweek = row['time_stamp'].weekday()
        if lweek == 6 or lweek == 5:
            lweek = 0
        else:
            lweek = 1
        if ix < tshape[0] - 1 and ix > 2:
            nexttrain = train1.ix[ix + 1]
            befortrain = train1.ix[ix - 1]
            if row['user_id'] == nexttrain['user_id'] or row['user_id'] == befortrain['user_id']:
                flagtwice = 1
            else:
                flagtwice = 0
        else:
            flagtwice = 0
        if lday < 60:
            flagtime = 0
        elif lday > 90:
            flagtime = 2
        else:
            flagtime = 1
        # 增加位置
        newlon, newlat = millerxy(row['longitude'], row['latitude'])
        addfeature.append([lday, lweek, flagtwice, flagtime, newlon, newlat])

    addfea = pd.DataFrame(np.array(addfeature), index=[x for x in range(tshape[0])],
                          columns=['lday', 'lweek', 'flagtwice', 'flagtime', 'newlon', 'newlat'])
    train2 = pd.concat([train1, addfea], axis=1)
    return train2


def addfea2(train1, shoplist):
    '''统计练过每个店铺时的位置，聚类出一个中心，然后用这个中心当做中心点，统计位置与中心点的距离，设定阈值并二值化'''
    df_train = train1[train1.shop_id.notnull()]
    df_test = train1[train1.shop_id.isnull()]
    addfeature = []
    # 训练集增加位置属性
    shop = pd.read_csv(path + u'训练数据-ccf_first_round_shop_info.csv')
    shop = shop[shop.mall_id == df_train['mall_id'][0]]
    shopdic = {}
    for i in shoplist:
        tmp1 = shop[shop.shop_id == i]
        shopdic[i] = [tmp1['longitude'].values[0], tmp1['latitude'].values[0]]

    posit_sit = dict(zip(shoplist, [[] for x in range(len(shoplist))]))
    for index, row in df_train.iterrows():
        # wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        posit_sit[row['shop_id']].append([row['longitude'], row['latitude']])
    for i in posit_sit:
        tmp1 = posit_sit[i]
        if len(tmp1) == 1:
            shopdic[i] = tmp1[0]
        else:
            shopdic[i] = KMeans(n_clusters=1).fit(posit_sit[i]).cluster_centers_[0].tolist()

    for ix, row in df_train.iterrows():
        tmp3 = []
        for i in shoplist:
            if geodistance(shopdic[i][0], shopdic[i][1], row['longitude'], row['latitude']) < 50:
                tmp3 = tmp3 + [True]
            else:
                tmp3 = tmp3 + [False]
                # tmp3 = tmp3 + [geodistance(shopdic[i][0], shopdic[i][1], row['longitude'], row['latitude'])]
        addfeature.append(tmp3)

    # 测试集增加位置属性
    for ix, row in df_test.iterrows():
        tmp3 = []
        for i in shoplist:
            if geodistance(shopdic[i][0], shopdic[i][1], row['longitude'], row['latitude']) < 50:
                tmp3 = tmp3 + [True]
            else:
                tmp3 = tmp3 + [False]
                # tmp3 = tmp3 + [geodistance(shopdic[i][0], shopdic[i][1], row['longitude'], row['latitude'])]
        addfeature.append(tmp3)

    addfea = pd.DataFrame(np.array(addfeature), index=[x for x in range(train1.shape[0])],
                          columns=['dis' + x for x in shoplist])
    train2 = pd.concat([train1, addfea], axis=1)
    return train2
    # xgbtrain2 = xgb.DMatrix(train2[feature].head(int(tshape[0] * 0.01)), train2['label'].head(int(tshape[0] * 0.01)))


def addlabel(train1):
    '''生成对应的标签店铺列表'''
    df_train = train1[train1.shop_id.notnull()]
    df_test = train1[train1.shop_id.isnull()]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
    # num_class = df_train['label'].max() + 1
    listshop = list(lbl.classes_)
    train2 = df_train.append(df_test, ignore_index=True)
    return train2, listshop


def mean_confidence_interval(data, confidence=0.95):
    '''95%置信区间的数据,m为均值，m-h，m+h为置信区间下限和上限'''
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    outdata = (np.array(data) - (m - h)) / (2 * h)
    return [outdata, m, h]


if __name__ == '__main__':
    allfile = os.listdir('../data')
    result = pd.DataFrame()
    bestscore = []
    kk = 0
    for infile in allfile:
        with open('../data/' + infile, 'rb') as f:
            m = pickle.load(f)

        m_2, mshop = addlabel(m)  #
        m_21 = addfea(m_2)  #
        m_3 = addfea2(m_21, mshop)
        m_3['newlon'] = mean_confidence_interval(m_3['newlon'], confidence=0.95)[0]
        m_3['newlat'] = mean_confidence_interval(m_3['newlat'], confidence=0.95)[0]
        m_3['lweek'] = m_3['lweek'].astype('bool')
        m_3['flagtwice'] = m_3['flagtwice'].astype('bool')
        m_3['lday'] = m_3['lday'].astype('bool')
        m_3['lday'] = m_3['lday'].astype('int')
        with open('../data2/m' + infile, 'wb') as f:
            pickle.dump(m_3, f)
        with open('../data2/mshop' + infile, 'wb') as f:
            pickle.dump(mshop, f)
        kk = kk + 1
        print(kk, '个完成', infile)

    print('f')

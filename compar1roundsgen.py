# -*- coding: utf-8 -*-
"""
对比文件特征
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


def crover(inparams):
    '''xgboost训练param网格搜索
    inparams：输入词典paramscans（需至少有一项参数为列表）
    alldic：输出每种网格搜索的参数
    outvar：输入变量是哪种
    '''
    listpara = []
    invir = []
    virdic = {}
    alldic = []
    outvar = []
    for i in inparams:
        if isinstance(inparams[i], type([])):
            listpara.append(inparams[i])
            invir.append(i)
        else:
            virdic[i] = inparams[i]
    if len(listpara) == 0:
        print('输入不是列表')
        os._exit()
    listpara2 = list(itertools.product(*listpara))
    for i in listpara2:
        outdic = virdic.copy()
        outlist = []
        for j in range(len(invir)):
            outdic[invir[j]] = i[j]
            outlist.append(i[j])
        # outvar.append()
        # print(outdic,outlist)
        # yield outdic,outlist
        alldic.append(outdic)
        outvar.append(outlist)
    return alldic, outvar


def parti(df_train, feature, testradio, count):
    '''交叉验证
    df_train：输入数据集
    feature：数据列类型
    testradio：测试集占比
    count：取第几部分当测试集
    比如testradio, count分别为0.2,1，则为取前面20%作为测试集'''
    lendata = df_train.shape[0]
    begin = int(lendata * testradio * (count - 1))
    end = int(lendata * testradio * count)
    test = df_train.iloc[begin:end, :]
    # train=pd.concat([df_train.iloc[:begin,:],df_train.iloc[end:,:] ], axis=1)
    train = df_train.iloc[:begin, :].append(df_train.iloc[end:, :])
    return xgb.DMatrix(train[feature], train['label']), xgb.DMatrix(test[feature], test['label'])






def genfeature(intrain, fea):
    '''调整需要验证的属性来优化结果，featall1：wifi连接属性'''
    featall1 = []
    featall2 = fea
    for i in intrain.columns:
        if 'b_' in i:
            featall1.append(i)
    feature = featall1 + featall2
    return feature


# def trainxgb(train1, feature):
#     df_train = train1[train1.shop_id.notnull()]
#     # df_test = train1[train1.shop_id.isnull()]
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(list(df_train['shop_id'].values))
#     df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
#     num_class = df_train['label'].max() + 1
#     # params = {
#     #     'objective': 'multi:softmax',
#     #     'eta': 0.1,
#     #     'max_depth': 10,
#     #     'eval_metric': 'merror',
#     #     # 'seed': 0,
#     #     'missing': -999,
#     #     'num_class': num_class,
#     #     # 'nthread': 4,
#     #     #'min_child_weight': 1,
#     #     'subsample': 1,  # 随机采样训练样本
#     #     'colsample_bytree': 1,  # 生成树时进行的列采样
#     #     # 'seed': 1000,
#     #     'gamma': [0,  0.2],
#     #     'lambda': 1,
#     #     'silent': 1
#     # }
#     params = {
#         'objective': 'multi:softmax',
#         'eta': 0.1,
#         'max_depth': [8, 13],
#         'eval_metric': 'merror',
#         # 'seed': 0,
#         'missing': -999,
#         'num_class': num_class,
#         # 'nthread': 4,
#         # 'min_child_weight': [1,5],
#         'subsample': [0.5, 0.7],  # 随机采样训练样本
#         'colsample_bytree': [0.501, 0.701],  # 生成树时进行的列采样
#         # 'seed': 1000,
#         'gamma': 0.2,
#         'lambda': 1,
#         'silent': 1
#     }
#
#     tshape = df_train.shape
#
#     # xgbtrain2 = xgb.DMatrix(df_train[feature].head(int(tshape[0] * 0.3)), df_train['label'].head(int(tshape[0] * 0.3)))
#     # xgbtrain = xgb.DMatrix(df_train[feature].tail(int(tshape[0] * 0.7)), df_train['label'].tail(int(tshape[0] * 0.7)))
#
#     xgbtrain, xgbtrain2 = parti(df_train, feature, 0.3, 2)
#     # xgbtest = xgb.DMatrix(df_test[feature])
#     watchlist = [(xgbtrain, 'train'), (xgbtrain2, 'test')]
#     num_rounds = 400
#     allvar = []
#     allpara, listvar = crover(params)
#     for k in range(len(allpara)):
#         # print (i,j)
#         i = allpara[k]
#         j = listvar[k]
#         model = xgb.train(i, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
#         # allvar.append([model.best_score, model.best_iteration, i,j])
#         allvar.append([model.best_score, model.best_iteration] + j)
#         # print([xx[3] for xx in allvar])
#     # print(allvar)
#     for x in allvar:
#         print(x)
#         print('\n')
#     return allvar


def trainxgbcro(train1, feature, result):
    '''带交叉验证的训练：网格搜索params里面的参数，使用交叉验证最好的作为最后仿真的参数'''
    df_train = train1[train1.shop_id.notnull()]
    df_test = train1[train1.shop_id.isnull()]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))
    num_class = df_train['label'].max() + 1
    # params = {
    #     'objective': 'multi:softmax',
    #     'eta': 0.1,
    #     'max_depth': 10,
    #     'eval_metric': 'merror',
    #     # 'seed': 0,
    #     'missing': -999,
    #     'num_class': num_class,
    #     # 'nthread': 4,
    #     #'min_child_weight': 1,
    #     'subsample': 1,  # 随机采样训练样本
    #     'colsample_bytree': 1,  # 生成树时进行的列采样
    #     # 'seed': 1000,
    #     'gamma': [0,  0.2],
    #     'lambda': 1,
    #     'silent': 1
    # }
    params = {
        'objective': 'multi:softmax',
        'eta': 0.08,
        'max_depth': 10,
        'eval_metric': 'merror',
        # 'seed': 0,
        'missing': -999,
        'num_class': num_class,
        # 'nthread': 4,
        # 'min_child_weight': [1,5],
        'subsample': [0.5, 0.7],  # 随机采样训练样本
        'colsample_bytree': [0.301, 0.501],  # 生成树时进行的列采样!!!!!!![0.301,0.501,0.701]
        # 'seed': 1000,
        # 'gamma': [1e-5, 1e-2, 0.1, 1, 100],
        # 'lambda': [1e-5, 1e-2, 0.1, 1, 100],
        'silent': 1
    }

    tshape = df_train.shape

    # xgbtrain2 = xgb.DMatrix(df_train[feature].head(int(tshape[0] * 0.3)), df_train['label'].head(int(tshape[0] * 0.3)))
    # xgbtrain = xgb.DMatrix(df_train[feature].tail(int(tshape[0] * 0.7)), df_train['label'].tail(int(tshape[0] * 0.7)))

    xgbtrain, xgbtrain2 = parti(df_train, feature, 0.3, 2)
    # xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [(xgbtrain, 'train'), (xgbtrain2, 'test')]
    num_rounds = 400
    allvar = []
    paralist = []
    allpara, listvar = crover(params)
    for k in range(len(allpara)):
        # print (i,j)
        i = allpara[k]
        j = listvar[k]
        paralist.append(i)
        meanscor = 0
        meaniter = 0
        meanscor = []
        for cr in [1, 3, 5]:
            xgbtrain, xgbtrain2 = parti(df_train, feature, 0.2, cr)
            watchlist = [(xgbtrain, 'train'), (xgbtrain2, 'test')]
            model = xgb.train(i, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
            # meanscor=meanscor+model.best_score
            meanscor.append(model.best_score)
            meaniter = meaniter + model.best_iteration
        # allvar.append([model.best_score, model.best_iteration, i,j])
        allvar.append([meanscor, sum(meanscor) / len(meanscor), meaniter / 3] + j)
        # print([xx[3] for xx in allvar])
    allvar2 = np.array(allvar)
    tmpsit = [x[0] for x in allvar]
    params = paralist[allvar2[:, 1].argmin()]

    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
    num_rounds = int(allvar[allvar2[:, 1].argmin()][2]) + 10
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
    # print('本次迭代最优值和参数为',model.best_score,listvar[tmpsit.index(min(tmpsit))])

    df_test['label'] = model.predict(xgbtest)
    df_test['shop_id'] = df_test['label'].apply(lambda x: lbl.inverse_transform(int(x)))
    r = df_test[['row_id', 'shop_id']]
    result = pd.concat([result, r])
    result['row_id'] = result['row_id'].astype('int')
    result.to_csv('jcsub.csv', index=False)
    bestscore.append([infile, model.best_iteration, model.best_score, allvar[tmpsit.index(min(tmpsit))][3],
                      allvar[tmpsit.index(min(tmpsit))]])

    with open('jcbest.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(bestscore)

    return result, bestscore


if __name__ == '__main__':
    # mall='m_6167'
    allfile = os.listdir('../data')
    result = pd.DataFrame()
    bestscore = []
    kk = 0
    for infile in allfile:
        with open('../data2/m' + infile, 'rb') as f:
            m_3 = pickle.load(f)
        with open('../data2/mshop' + infile, 'rb') as f:
            mshop = pickle.load(f)
        feature = genfeature(m_3,
                             ['newlat', 'newlon', 'lweek',
                              'lday'])  # , 'flagtwice', 'lday'])# + ['dis' + x for x in mshop]
        result, bestscore = trainxgbcro(m_3, feature, result)
        kk = kk + 1
        print(kk, '个完成', infile)
    print('f')

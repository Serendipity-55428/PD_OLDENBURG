#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: data_submaking
@time: 2019/8/28 下午4:41
'''
import numpy as np
import pandas as pd
from inputdata import LoadFile, SaveFile, guiyi,onehot
from collections import Counter
from data_making import fft_transformer

def making2(p, argchoice_r):
    '''
    选择特殊半径组成数据集制作数据
    :param p: 数据集路径, str
    :param argchoice_r: 半径列表, list
    :return: 子数据集
    '''
    dataset = LoadFile(p)
    dataset_pd = pd.DataFrame(data=dataset, columns=['f'+'%s' % i for i in range(dataset.shape[-1]-1)]+['label'])
    rs = dataset[:, -1]
    rs_dict = Counter(rs)
    rs_list = list(rs_dict.keys())
    rs_list = sorted(rs_list)
    choice_r = list(filter(lambda x:rs_list.index(x) in argchoice_r, rs_list))
    #统计所有半径值数据中数据量最大的数据量值
    max = 0
    for i in choice_r:
        data_subpd = dataset_pd.loc[dataset_pd['label'] == i]
        if data_subpd.values.shape[0] > max:
            max = data_subpd.values.shape[0]
    data_all = np.array([0])
    data_array_con = np.zeros(dtype=np.float32, shape=[1])
    for i in choice_r:
        data_subpd = dataset_pd.loc[dataset_pd['label'] == i]
        # print(data_subpd.values.shape)
        num = max // data_subpd.values.shape[0]
        num = num if max % data_subpd.values.shape[0] == 0 else num+1
        data_array = data_subpd.values
        np.random.shuffle(data_array)
        for i in range(num):
            if data_array_con.any() == 0:
                data_array_con = data_array
            else:
                data_array_con = np.vstack((data_array_con, data_array))
        print(data_array_con.shape, max)
        data_all = data_array_con[:max, :] if data_all.any() == 0 else np.vstack((data_all, data_array_con[:max, :]))
        data_array_con = np.array([0])
    return data_all

if __name__ == '__main__':
    p1 = r'/home/xiaosong/桌面/OLDENBURG_sub1.pickle'
    p2 = r'/home/xiaosong/桌面/OLDENBURG_sub2.pickle'
    p = r'/home/xiaosong/桌面/OLDENBURG_all.pickle'
    argchoice_r1 = [7, 8, 11, 14]
    argchoice_r2 = [1, 2, 3, 4, 5, 6, 9, 10, 12, 13]
    # data_all1 = making2(p=p, argchoice_r=argchoice_r1)
    # dataset_4feature, dataset_dense, label = data_all1[:, :4], data_all1[:, 4:-1], data_all1[:, -1][:,np.newaxis]
    # dataset_fft = fft_transformer(dataset_dense, 100)
    # dataset = np.hstack((dataset_4feature, dataset_fft, label))
    # dataset_guiyi_1 = guiyi(dataset)
    # print(dataset_guiyi_1.shape)
    # print(Counter(dataset_guiyi_1[:, -1]))
    # print('onehot编码后结果：')
    # one_hot = onehot(dataset_guiyi_1)
    # print(one_hot.shape)
    # print(np.sum(one_hot[:, -4:], axis=0))
    # print(np.min(dataset_guiyi_1, axis=0))
    # SaveFile(data=dataset_guiyi_1, savepickle_p=p1)

    data_all2 = making2(p=p, argchoice_r=argchoice_r2)
    dataset_4feature, dataset_dense, label = data_all2[:, :4], data_all2[:, 4:-1], data_all2[:, -1][:, np.newaxis]
    dataset_fft = fft_transformer(dataset_dense, 100)
    dataset = np.hstack((dataset_4feature, dataset_fft, label))
    dataset_guiyi_2 = guiyi(dataset)
    print(dataset_guiyi_2.shape)
    # print('onehot编码后结果: ')
    # one_hot2 = onehot(dataset_guiyi_2)
    # print(one_hot2.shape)
    # print(np.sum(one_hot2[:, -10:], axis=0))
    # print(np.min(dataset_guiyi_2, axis=0))
    SaveFile(data=dataset_guiyi_2, savepickle_p=p2)

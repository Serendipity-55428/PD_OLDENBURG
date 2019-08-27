#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: data_making
@time: 2019/8/27 下午5:08
'''
import numpy as np
import pandas as pd
from collections import Counter
from inputdata import LoadFile, SaveFile, guiyi, onehot
from datagenerator import checkclassifier

def making(nums_cl, dataset):
    '''
    按照输入的每个半径的分组以及数据量个数划分数据集
    :param nums_cl: 每个半径数据的数量以及划分为哪一类，输入时按照最优半径从大到小排列 type=[[nums,cl], [,], ...]
    :param dataset: 数据集/最后一列为标签
    :return: 经过处理后的数据集,数据集最后一列为标签
    '''
    dataset_output = np.zeros(shape=[1, dataset.shape[-1]])
    #最优半径从小到大排序
    dict_r = Counter(dataset[:, -1])
    # print(dict_r)
    dataset_2 = dict_r.keys()
    # print(dataset_2)
    dataset_2 = np.array(list(dataset_2))
    # print(dataset_2)
    r_sort = np.sort(dataset_2)
    dataset_pd = pd.DataFrame(data=dataset, columns=['f'+'%s' % i for i in range(dataset.shape[-1]-1)]+['label'])
    i = 0
    for r in r_sort:
        print('正在执行半径%s' % r)
        #定义用于拼接的临时矩阵
        data_per_pd = dataset_pd.loc[dataset_pd['label'] == r]
        data_per = np.array(data_per_pd)
        np.random.shuffle(data_per)
        #获取该最优半径所需要的数据量和标签值
        nums, cl = nums_cl[i]
        data_per = np.hstack((data_per[:nums, :-1], np.ones(dtype=np.float32, shape=[nums, 1])*cl))
        dataset_output = data_per if dataset_output.any() == 0 else np.vstack((dataset_output, data_per))
        i += 1
    return dataset_output

def fft_transformer(dataset, N):
    '''
    对矩阵中各行按照指定点数做FFT变换
    :param dataset: 待处理矩阵
    :param N: 变换后点数
    :return: 处理后矩阵
    '''
    fft_abs = np.abs(np.fft.fft(a=dataset, n=N, axis=1))
    return fft_abs

if __name__ == '__main__':
    p = r'/home/xiaosong/桌面/OLDENBURG_all.pickle'
    dataset = LoadFile(p)
    nums_cl = [[6557, 0], [611, 2], [101, 2], [13, 2], [554, 2], [155, 2], [100, 2], [1165, 1], [1993, 1], [947, 2],
               [1133, 2], [1152, 1], [542, 2], [754, 2], [2163, 1]]
    dataset_output = making(nums_cl=nums_cl, dataset=dataset)
    print(dataset_output.shape)
    checkclassifier(dataset_output[:, -1])
    # SaveFile(dataset_output, savepickle_p=r'/home/xiaosong/桌面/OLDENBURG_3cl.pickle')
    dataset_4feature, dataset_dense, label = dataset_output[:, :4], dataset_output[:, 4:-1], dataset_output[:, -1][:, np.newaxis]
    dataset_fft = fft_transformer(dataset_dense, 100)
    dataset = np.hstack((dataset_4feature, dataset_fft, label))
    dataset_guiyi = guiyi(dataset)
    print(dataset_guiyi.shape)
    # print(np.min(dataset_guiyi, axis=0))
    SaveFile(data=dataset_guiyi, savepickle_p=r'/home/xiaosong/桌面/OLDENBURG_3cl.pickle')
    dataset_onehot = onehot(dataset_guiyi)
    print(np.sum(dataset_onehot[:, -3:], axis=0))

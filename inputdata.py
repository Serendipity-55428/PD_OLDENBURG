#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: inputdata
@time: 2019/8/27 下午3:46
'''
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import os
def LoadFile(p):
    '''
    读取文件
    :param p: 数据集绝对路径
    :return: 数据集
    '''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

def SaveFile(data, savepickle_p):
        '''
        存储整理好的数据
        :param data: 待存储数据
        :param savepickle_p: pickle后缀文件存储绝对路径
        :return: None
        '''
        if not os.path.exists(savepickle_p):
            with open(savepickle_p, 'wb') as file:
                pickle.dump(data, file)

def onehot(dataset):
    '''
    将所有标签按数值大小编码为one-hot稀疏向量
    :param dataset: 数据集特征矩阵,最后一列为半径标签
    :return: 标签半径被编码后的数据集
    '''
    #将label_dict排列
    label_pri = dataset[:, -1]
    label_dict = Counter(label_pri)
    #标签生序字典
    label_sort = sorted(label_dict.items(), key=lambda l: l[0])
    label_pri_pd = pd.DataFrame(data=dataset, columns=[str(i) for i in range(dataset.shape[-1])])
    #数据集按标签生序排列
    label_pri_pd.sort_values('%s' % label_pri_pd.columns[-1], inplace=True)
    label_pri_sort = np.array(label_pri_pd)
    index = 0
    one_hot_metric = np.zeros(dtype=np.float32, shape=(dataset.shape[0], len(label_sort)))
    for group in label_sort:
        one_hot_metric[index:index+group[-1], label_sort.index(group)] = 1
        index += group[-1]
    dataset_onehot = np.hstack((label_pri_sort[:, :-1], one_hot_metric))
    np.random.shuffle(dataset_onehot)
    return dataset_onehot

def train_testspliting(dataset, train_path, test_path):
    '''
    7/3分训练集和测试集
    :param dataset: 原始数据集/单列标签
    :param train_path: 训练集存储路径
    :param test_path: 测试机存储路径
    :return: None
    '''
    rng = np.random.RandomState(0)
    rng.shuffle(dataset)
    test_set = dataset[:4500, :]
    train_set = dataset[4500:, :]
    SaveFile(data=train_set, savepickle_p=train_path)
    SaveFile(data=test_set, savepickle_p=test_path)

def input(dataset, batch_size):
    '''
    按照指定批次大小随机输出训练集中一个批次的特征/标签矩阵
    :param dataset: 数据集特征矩阵(特征经过01编码后的)
    :param batch_size: 批次大小
    :return: 特征矩阵/标签
    '''
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size, :]

def spliting(dataset, size):
    '''
    留一法划分训练和测试集
    :param dataset: 特征数据集/标签
    :param size: 测试集大小
    :return: 训练集和测试集特征矩阵/标签
    '''
    #随机得到size大小的交叉验证集
    test_row = np.random.randint(low=0, high=len(dataset)-1, size=(size))
    #从dataset中排除交叉验证集
    train_row = list(filter(lambda x: x not in test_row, range(len(dataset)-1)))
    return dataset[train_row, :], dataset[test_row, :]

def guiyi(dataset):
    '''
    对带标签的数据集进行特征归一化
    :param dataset: 带标签的数据集
    :return: 归一化后的特征/标签矩阵
    '''
    feature_min = np.min(dataset[:, :-1], axis=0)
    feature_max = np.max(dataset[:, :-1], axis=0)
    feature_guiyi = (dataset[:, :-1] - feature_min) / (feature_max - feature_min)
    dataset_guiyi = np.hstack((feature_guiyi, dataset[:, -1][:, np.newaxis]))
    return dataset_guiyi

if __name__ == '__main__':
    rng = np.random.RandomState(0)
    p = '/home/xiaosong/桌面/PNY_fft_dadta_bigclass.pickle'
    with open(p, 'rb') as f:
        data = pickle.load(f)
    p1 = '/home/xiaosong/桌面/PNY_train.pickle'
    p2 = '/home/xiaosong/桌面/PNY_test.pickle'
    #划分训练集和测试集
    # train_testspliting(dataset=data, train_path=p1, test_path=p2)
    # data1 = LoadFile(p1)
    # print(data1.shape)
    # data2 = LoadFile(p2)
    # print(data2.shape)
    sta = Counter(data[:, -1])
    print(sta)
    sta_1 = onehot(data)
    sta_1 = sta_1[:, -4:]
    sta_1 = np.sum(sta_1, axis=0)
    print(sta_1)
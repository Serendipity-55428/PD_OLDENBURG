#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: datagenerator
@time: 2019/8/27 下午3:43
'''
import os.path
import pickle
import numpy as np
import xlwt
import pandas as pd
import bisect
from collections import Counter
from inputdata import LoadFile, SaveFile

def GeneratorData(type, filename_prefix):
    '''
    生成numpy矩阵数据
    :param type: 1/2(1:'PNY'/2:'OLDENBURG')
    :param filename_prefix: 所有待处理文件绝对路径前缀
    :return: numpy数据矩阵
    '''
    movingObjects = [1000, 1500, 2000, 4000, 6000, 8000, 10000, 20000, 40000, 60000, 80000, 100000]
    maxSpeed = ['v' + '%s' % i for i in range(1, 7)]
    maxSpeed = list(zip(maxSpeed, [143.022339, 143.022339, 28.593955, 28.593955, 5.703022, 5.703022]))
    friends = [5, 10, 20, 30, 40]
    Threshold = [i for i in range(1, 11)]
    #初始化待输出numpy数据矩阵
    dataset = np.zeros(shape=(1, 25))
    for file_movingObjects in movingObjects:
        for file_maxSpeed in maxSpeed:  # file_maxSpeed类型为元组 #改
            for file_friends in friends:
                for file_Threshold in Threshold:
                    # 前4个特征列表，需要和文件中20个特征、1个最优半径结合后转换为ndarray类型向量
                    sub_feature = [file_movingObjects, file_maxSpeed[-1], file_friends, file_Threshold]
                    # 数据文件命名格式：type_movingObjects_maxSpeed_friends_Threshold
                    p = filename_prefix + '/' + r'%s_%s_%s_%s_%s.dat' % \
                        (type, file_movingObjects, file_maxSpeed[0], file_friends, file_Threshold)
                    if os.path.exists(p):
                        with open(p, 'r') as f:
                            # 将文件转为字符串类型，切片可以自动消除‘\n'字符
                            file = f.readlines()
                            # print(file[2][-2])
                            for ts in range(5):
                                # 设置在各个r组内指向每一个cost的指针
                                inside_point = 5 + ts * 3
                                # 定义全局指针
                                point = inside_point
                                # 初始化存储每20倍数时刻最小代价值索引,初始化指向point所指向的位置
                                min_cost = point
                                while point <= 255:
                                    if float(file[point][7:-1]) <= float(file[min_cost][7:-1]):
                                        min_cost = point
                                    # 指向下一个r中同一时间的cost
                                    point += 17

                                # min_cost位置回退2便指向对应20个密度特征，回退(ts + 1) * 3个便指向当前最优半径
                                twenty_feature = file[min_cost - 2][:-2]
                                twenty_feature = twenty_feature.split(' ')
                                twenty_feature = [float(i) for i in twenty_feature] if len(
                                    [float(i) for i in twenty_feature]) == 20 else \
                                    [float(i) for i in twenty_feature[1:]]
                                # 拼接特征向量
                                sub_feature.extend(twenty_feature)
                                sub_feature.append(float(file[min_cost - (ts + 1) * 3][4:-1]))
                                # print(len(sub_feature))
                                dataset = np.array(sub_feature) if dataset.any() == 0 else \
                                    np.vstack((dataset, np.array(sub_feature)[np.newaxis, :]))

                                # 重置sub_feature列表为前4个特征
                                sub_feature = [file_movingObjects, file_maxSpeed[-1], file_friends, file_Threshold]

    return dataset

def Numpy2Excel(data, save_p, name= 'PNY'):
    '''
    将numpy数组转化为xls/xlsx
    :param data: 待转化数据
    :param save_p: 输出xls/xlsx绝对路径
    :return: xls/xlsx
    '''
    file = xlwt.Workbook()
    table = file.add_sheet(sheetname= name, cell_overwrite_ok=True)
    # 建立列名:
    table.write(0, 0, 'movingObjects')
    table.write(0, 1, 'maxSpeed')
    table.write(0, 2, 'friends')
    table.write(0, 3, 'Threshold')
    for i in range(1, 21):
        table.write(0, i+3, 'Density %s' % i)
    table.write(0, 24, 'optimal_r')

    for h in range(data.shape[0]):
        for v in range(data.shape[-1]):
            table.write(h+1, v, str(data[h, v]))

    file.save(save_p)


def checkclassifier(vector):
    '''
    对输入数据向量进行各个数量类别统计
    :param vector: 待统计数据向量
    :return: None
    '''
    statistic = Counter(vector)
    statistic = list(statistic.items())
    statistic.sort(key=lambda i: i[0])
    for key, value in statistic:
        print('%s: %s' % (key, value))
    # print('\n')

# 类别划分通用函数
def transform(label):
    '''
    将回归标签转换为类别标签
    :param label: 待转换回归标签
    :return: 类别标签
    '''
    def divide(x):
        divide_point = [10, 20, 100, 300] #划分半径大类别标签
        divide_label = [i for i in range(4)]
        position = bisect.bisect(a=divide_point, x=x)
        return divide_label[position]
    divide_ufunc = np.frompyfunc(divide, 1, 1)
    return divide_ufunc(label)

#制作六类标签
def big_classify(dataset, func):
    '''
    大类标签制作器
    :param dataset: 数据集/标签
    :param func: 大分类函数
    :return: 根据大类分割后的数据集, 标签为单列数字
    '''
    #提取标签
    label = dataset[:, -1]
    #生成类别标签
    label_class = func(label)
    #合成新数据
    new_dataset = np.hstack((dataset[:, :-1], label_class[:, np.newaxis]))
    print(Counter(label_class))
    return new_dataset

if __name__ == '__main__':
    rng = np.random.RandomState(2)
    #生成oldenburg数据
    type= 'oldenburg'
    filename_prefix = r'/home/xiaosong/oldenburg'
    dataset = GeneratorData(type= type, filename_prefix= filename_prefix)
    rng.shuffle(dataset)
    #查看数据中各个半径的数量
    checkclassifier(dataset[:, -1])

    #检查最优半径为0.01的数量
    zero_r, nonzero_r = 0, 0
    r = dataset[:, -1]
    for i in r:
        if i == 0.01:
            zero_r += 1
    nonzero_r = len(r) - zero_r
    print('最优半径为0.01个数: %s \n最优半径不为0.01的个数: %s' % (zero_r, nonzero_r))
    # print(dataset)
    print(dataset.shape)
    print(dataset.dtype)

    #保存文件到.xlsx/.pickle
    save_pickle = r'/home/xiaosong/桌面/OLDENBURG_all.pickle'
    # save_xlsx = r'/home/xiaosong/桌面/datasets.xlsx'
    # Numpy2Excel(data= dataset, save_p= save_xlsx)
    SaveFile(data=dataset, savepickle_p=save_pickle)

    #划分最优半径含0.01和不含0.01的数据
    # data_all_ = LoadFile(p=save_pickle)
    # rs = data_all_[:, -1]
    # rs_dict = Counter(rs)
    # for key, value in rs_dict.items():
    #     print('%s: %s' % (key, value))
    # rs_list = list(rs_dict.keys())
    # rs_list = sorted(rs_list)
    # for key, value in dict(list(enumerate(rs_list))).items():
    #     print(key, value)

    # data_all = pd.DataFrame(data_all_)
    # print(data_all)
    # data_no_noise = data_all.loc[data_all[24] != 0.01]
    # data_noise = data_all.loc[data_all[24] == 0.01]
    # data_no_noise = np.array(data_no_noise)
    # data_noise = np.array(data_noise)
    # print(np.array(data_no_noise).shape, np.array(data_noise).shape)
    # SaveFile(data=data_noise, savepickle_p=r'/home/xiaosong/桌面/PNY_noise.pickle')
    # SaveFile(data=data_no_noise, savepickle_p=r'/home/xiaosong/桌面/PNY_no_noise.pickle')
    # data_noise = LoadFile(p=r'/home/xiaosong/桌面/PNY_noise.pickle')
    # data_no_noise = LoadFile(r'/home/xiaosong/桌面/PNY_no_noise.pickle')
    # print(data_noise.shape, data_no_noise.shape)

    #制作数量为10000的数据集，无噪声数据集数量不够可以用少量噪声数据集进行填充
    # data_train = np.vstack((data_no_noise, data_noise[:(10000-data_no_noise.shape[0]), :]))
    # rng.shuffle(data_train)
    # SaveFile(data=data_train, savepickle_p=r'/home/xiaosong/桌面/PNY_data_train.pickle')
    # print(data_train.shape)
    # statistic = Counter(data_train[:, -1])
    # for key, value in statistic.items():
    #     print('%s: %s' % (key, value))

    #制作平均密度序列的fft变换膜值序列
    # data_fft_noise = np.hstack((data_noise[:, :4], np.abs(np.fft.fft(a=data_noise[:, 4:-1], n=100, axis=1)), data_noise[:, -1][:, np.newaxis]))
    # data_fft_no_noise = np.hstack((data_no_noise[:, :4], np.abs(np.fft.fft(a=data_no_noise[:, 4:-1], n=100, axis=1)), data_no_noise[:, -1][:, np.newaxis]))
    # print(data_fft_noise.shape, data_fft_no_noise.shape)
    # SaveFile(data=data_fft_noise, savepickle_p=r'/home/xiaosong/桌面/PNY_fft_noise.pickle')
    # SaveFile(data=data_fft_no_noise, savepickle_p=r'/home/xiaosong/桌面/PNY_fft_no_noise.pickle')

    #制作数量为10000的fft变换取模后的数据集
    # data_fft = np.vstack((data_fft_no_noise, data_fft_noise[:(10000-data_fft_no_noise.shape[0]), :]))
    # rng.shuffle(data_fft)
    # SaveFile(data=data_fft, savepickle_p=r'/home/xiaosong/桌面/PNY_data_fft.pickle')
    # print(data_fft.shape)

    #制作大类标签数据集
    # data_fft = LoadFile(p=r'/home/xiaosong/桌面/PNY_data_fft.pickle')
    # fft_data_bigclass = big_classify(dataset=data_fft, func=transform)
    # SaveFile(data=fft_data_bigclass, savepickle_p=r'/home/xiaosong/桌面/PNY_fft_dadta_bigclass.pickle')




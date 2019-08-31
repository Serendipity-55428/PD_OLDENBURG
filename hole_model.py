#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: hole_model
@time: 2019/8/29 下午8:20
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from Saving_model_pb import SaveImport_model, use_pb, import_ops
from inputdata import LoadFile, guiyi
from data_making import fft_transformer

def first_check(input):
    '''
    第一层分类器
    :param input: 输入特征向量(矩阵)/标签
    :return: 半径所属大类别, numpy.ndarray
    '''
    pb_file_path = r'/home/xiaosong/桌面/oldenburg相关数据/model/full_model'
    g1 = tf.Graph()
    with g1.as_default():
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g1) as sess:
        sess.run(init)
        use_pb(sess_new=sess, pb_file_path=pb_file_path, file_suffix=r'/first_classifier')
        x_f = import_ops(sess_new=sess, op_name='placeholder/x_f')
        x_l = import_ops(sess_new=sess, op_name='placeholder/x_l')
        is_training = import_ops(sess_new=sess, op_name='placeholder/is_training')
        output = import_ops(sess_new=sess, op_name='dnn/Softmax')
        r_classify = sess.run(output, feed_dict={x_f: input[:, :4], x_l: input[:, 4:-1],
                                                 is_training: False})
        r_classify = np.argmax(a=r_classify, axis=1)
    return r_classify

def second_check1(input):
    '''
    第二层分类器1
    :param input: 输入特征向量(矩阵)/标签
    :return: 具体类别, numpy.ndarray
    '''
    dict1 = {0:7.395086, 1:14.790171, 2:29.580343, 3:51.7656}
    pb_file_path = r'/home/xiaosong/桌面/oldenburg相关数据/model/full_model_1'
    g2 = tf.Graph()
    with g2.as_default():
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g2) as sess:
        sess.run(init)
        use_pb(sess_new=sess, pb_file_path=pb_file_path, file_suffix=r'/second_classifier_1')
        x_f = import_ops(sess_new=sess, op_name='placeholder/x_f')
        x_l = import_ops(sess_new=sess, op_name='placeholder/x_l')
        is_training = import_ops(sess_new=sess, op_name='placeholder/is_training')
        output = import_ops(sess_new=sess, op_name='sub_dnn/Softmax')
        r_classify = sess.run(output, feed_dict={x_f: input[:, :4], x_l: input[:, 4:-1],
                                                 is_training: False})
        r_classify = np.argmax(a=r_classify, axis=1)
        #按照dict1中的映射将类别标记替换为实际半径
        r_finally = []
        for i in range(r_classify.shape[0]):
            r_finally.append(dict1[r_classify[i]])
        return r_finally

def second_check2(input):
    '''
    第二层分类器1
    :param input: 输入特征向量(矩阵)/标签
    :return: 具体类别, numpy.ndarray
    '''
    dict2 = {0:0.05, 1:0.672281, 2:0.924386, 3:1.479017, 4:3.697543, 5:4.930057,
             6:20.0, 7:22.185257, 8:36.975428, 9:44.370514}
    pb_file_path = r'/home/xiaosong/桌面/oldenburg相关数据/model/full_model_2'
    g3 = tf.Graph()
    with g3.as_default():
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g3) as sess:
        sess.run(init)
        use_pb(sess_new=sess, pb_file_path=pb_file_path, file_suffix=r'/second_classifier_2')
        x_f = import_ops(sess_new=sess, op_name='placeholder/x_f')
        x_l = import_ops(sess_new=sess, op_name='placeholder/x_l')
        is_training = import_ops(sess_new=sess, op_name='placeholder/is_training')
        output = import_ops(sess_new=sess, op_name='sub_dnn/Softmax')
        r_classify = sess.run(output, feed_dict={x_f: input[:, :4], x_l: input[:, 4:-1],
                                                 is_training: False})
        r_classify = np.argmax(a=r_classify, axis=1)
        # 按照dict2中的映射将类别标记替换为实际半径
        r_finally = []
        for i in range(r_classify.shape[0]):
            r_finally.append(dict2[r_classify[i]])
    return r_finally

def check(input):
    '''
    整体分类器
    :param input: 输入特征向量(矩阵)/标签
    :return: 最优半径矩阵, numpy.ndarray
    '''
    r_classify = first_check(input=input) #得到初级分类器的三类别标记, 分别为0, 1, 2
    # print(r_classify)
    #将初分类结果添加到输入数据集最后一列
    input_1 = pd.DataFrame(data=input, columns=[str(i) for i in range(input.shape[-1])])
    input_1['%s' % input.shape[-1]] = r_classify
    # print(input_1.values.shape)
    #筛选出经过初分类器后类别标记为0的所有数据
    input_firstc = input_1.loc[input_1['%s' % input.shape[-1]]==0]
    # print(input_firstc.values.shape)
    input_firstc = input_firstc.drop(columns=['%s' % input.shape[-1]])
    input_firstc = input_firstc.values
    # print(input_firstc.shape)
    r_finally1 = np.ones(dtype=np.float32, shape=[input_firstc.shape[0], 1]) * 0.01
    #筛选出经过初分类器后类别标记为1的所有数据
    input_secondc1 = input_1.loc[input_1['%s' % input.shape[-1]]==1]
    # print(input_secondc1.shape)
    input_secondc1 = input_secondc1.drop(columns=['%s' % input.shape[-1]])
    input_secondc1 = input_secondc1.values
    # print(input_secondc1.shape)
    #放入三分类器中进一步分类
    r_finally2 = np.array(second_check1(input_secondc1))[:, np.newaxis]
    #筛选出经过初分类器后类别标记为2的所有数据
    input_secondc2 = input_1.loc[input_1['%s' % input.shape[-1]]==2]
    input_secondc2 = input_secondc2.drop(columns=['%s' % input.shape[-1]])
    input_secondc2 = input_secondc2.values
    #放入11分类器中进一步分类
    r_finally3 = np.array(second_check2(input_secondc2))[:, np.newaxis]
    r_finally = np.vstack((r_finally1, r_finally2, r_finally3))
    r_real = np.vstack((input_firstc[:, -1][:, np.newaxis],
                        input_secondc1[:, -1][:, np.newaxis], input_secondc2[:, -1][:, np.newaxis]))
    return r_finally, r_real

if __name__ == '__main__':
    p = r'/home/xiaosong/桌面/OLDENBURG_all.pickle'
    input = LoadFile(p=p)
    np.random.shuffle(input)
    dataset_4feature, dataset_dense, label = input[:, :4], input[:, 4:-1], input[:, -1][:, np.newaxis]
    dataset_fft = fft_transformer(dataset_dense, 100)
    dataset = np.hstack((dataset_4feature, dataset_fft, label))
    dataset_guiyi_2 = guiyi(dataset)
    print(dataset_guiyi_2.shape)
    r_finally, r_real = check(input=dataset_guiyi_2[:5100, :])
    r_1 = np.where(np.abs(r_finally-r_real)<1e-2, 1, 0)
    r_sum = np.sum(r_1)
    acc = r_sum / r_1.shape[0]
    print('5100个测试样本的预测精确度为: %s' % acc)

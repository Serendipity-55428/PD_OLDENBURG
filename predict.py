#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: predict
@time: 2019/8/29 下午9:01
'''
import tensorflow as tf
import numpy as np
import os
from Saving_model_pb import SaveImport_model, use_pb, import_ops

def first_check(input):
    '''
    第一层分类器
    :param input: 输入特征向量(矩阵)/标签
    :return: 半径所属大类别, numpy.ndarray
    '''
    pb_file_path = r'/home/xiaosong/桌面/model/full_model'
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
    pb_file_path = r'/home/xiaosong/桌面/model/full_model_1'
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
    pb_file_path = r'/home/xiaosong/桌面/model/full_model_2'
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
    if r_classify == 0:
        r_finally = 0.01
    elif r_classify == 1:
        r_finally = second_check1(input=input)
    else:
        r_finally = second_check2(input=input)
    print('最优半径为: %s' % r_finally)
    return r_finally

if __name__ == '__main__':
    input = np.arange(105)[np.newaxis, :]
    r_finally = check(input=input)
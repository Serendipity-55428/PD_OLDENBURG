#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: graph2
@time: 2019/8/28 下午5:10
'''
import tensorflow as tf
import numpy as np
from inputdata import LoadFile, SaveFile, onehot, input, guiyi, spliting
from classifier_second import Resnet, subnet_resnet_1, subnet_resnet_2, subnet_cnnlstm_1, subnet_cnnlstm_2
from Saving_model_pb import SaveImport_model

def session_sub1(dataset_path, train_path='', test_path=''):
    '''
    节点连接
    :param dataset_path: 数据集路径
    :param train_path: 训练集数据路径,默认为空
    :param test_path: 测试集数据路径,默认为空
    :return: None
    '''
    #导入数据集
    dataset = LoadFile(p=dataset_path)
    dataset = onehot(dataset)
    g1 = tf.Graph()
    with g1.as_default():
        with tf.name_scope('placeholder'):
            x_f = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='x_f')
            x_l = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='x_l')
            y = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='y')
            learning_rate = tf.placeholder(dtype=tf.float32, name='lr')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        output = subnet_resnet_1(x_f=x_f, x_l=x_l, is_training=is_training)
        # output = subnet_cnnlstm_1(x_f=x_f, x_l=x_l, is_training=is_training)
        with tf.name_scope('prediction'):
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=output))
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.keras.backend.argmax(output, axis=1),
                                                  tf.keras.backend.argmax(y, axis=1)), tf.float32), name='pred')
        with tf.name_scope('etc'):
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g1) as sess:
        sess.run(init)
        # 划分训练集和测试集
        train_data, test_data = spliting(dataset, 2269)
        for i in range(1): #20000
            for data in input(dataset=train_data, batch_size=500):
                _ = sess.run(opt, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-4], y: data[:, -4:],
                                             learning_rate: 1e-2, is_training: False})
                if i % 100 == 0:
                    loss_ = sess.run(loss, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-4], y: data[:, -4:],
                                                      is_training: False})
                    acc_1 = sess.run(acc, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-4], y: data[:, -4:],
                                                     is_training: False})
            if i % 100 == 0:
                acc_2 = sess.run(acc, feed_dict={x_f: test_data[:, :4], x_l: test_data[:, 4:-4],
                                                 y: test_data[:, -4:], is_training: False})
                print('第%s轮训练集损失函数值为: %s  训练集准确率为: %s  测试集准确率为: %s' % (i, loss_, acc_1, acc_2))
        tf.summary.FileWriter('log/second_graph1', sess.graph)
        # 保存模型到文件当前脚本文件路径下的pb格式
        # saving_model = SaveImport_model(sess_ori=sess, file_suffix=r'/second_classifier_1',
        #                                 ops=(output, x_f, x_l, is_training), usefulplaceholder_count=4,
        #                                 pb_file_path=r'/home/xiaosong/桌面/model/full_model_1')
        # saving_model.save_pb()

def session_sub2(dataset_path, train_path='', test_path=''):
    '''
    节点连接
    :param dataset_path: 数据集路径
    :param train_path: 训练集数据路径,默认为空
    :param test_path: 测试集数据路径,默认为空
    :return: None
    '''
    #导入数据集
    dataset = LoadFile(p=dataset_path)
    # dataset = guiyi(dataset)
    dataset = onehot(dataset)
    g2 = tf.Graph()
    with g2.as_default():
        with tf.name_scope('placeholder'):
            x_f = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='x_f')
            x_l = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='x_l')
            y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')
            learning_rate = tf.placeholder(dtype=tf.float32, name='lr')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        output = subnet_resnet_2(x_f=x_f, x_l=x_l, is_training=is_training)
        # output = subnet_cnnlstm_2(x_f=x_f, x_l=x_l, is_training=is_training)
        with tf.name_scope('prediction'):
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=output))
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.keras.backend.argmax(output, axis=1),
                                                  tf.keras.backend.argmax(y, axis=1)), tf.float32), name='pred')
        with tf.name_scope('etc'):
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g2) as sess:
        sess.run(init)
        # 划分训练集和测试集
        train_data, test_data = spliting(dataset, 3300)
        for i in range(1): #20000
            for data in input(dataset=train_data, batch_size=500):
                _ = sess.run(opt, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-10], y: data[:, -10:],
                                             learning_rate: 1e-2, is_training: False})
                if i % 100 == 0:
                    loss_ = sess.run(loss, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-10], y: data[:, -10:],
                                                      is_training: False})
                    acc_1 = sess.run(acc, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-10], y: data[:, -10:],
                                                     is_training: False})
            if i % 100 == 0:
                acc_2 = sess.run(acc, feed_dict={x_f: test_data[:, :4], x_l: test_data[:, 4:-10],
                                                 y: test_data[:, -10:], is_training: False})
                print('第%s轮训练集损失函数值为: %s  训练集准确率为: %s  测试集准确率为: %s' % (i, loss_, acc_1, acc_2))
                if acc_2 > 0.95:
                    break
        tf.summary.FileWriter('log/second_graph2', sess.graph)
        # 保存模型到文件当前脚本文件路径下的pb格式
        # saving_model = SaveImport_model(sess_ori=sess, file_suffix=r'/second_classifier_2',
        #                                 ops=(output, x_f, x_l, is_training), usefulplaceholder_count=4,
        #                                 pb_file_path=r'/home/xiaosong/桌面/model/full_model_2')
        # saving_model.save_pb()

if __name__ == '__main__':
    p1 = r'/home/xiaosong/桌面/oldenburg相关数据/data_oldenburg/OLDENBURG_sub1.pickle'
    p2 = r'/home/xiaosong/桌面/oldenburg相关数据/data_oldenburg/OLDENBURG_sub2.pickle'
    session_sub1(dataset_path=p1)
    session_sub2(dataset_path=p2)
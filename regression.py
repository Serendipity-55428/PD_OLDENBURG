#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: regression
@time: 2019/8/31 下午1:57
'''
import tensorflow as tf
import numpy as np
import pandas as pd
from inputdata import LoadFile, SaveFile, onehot, input, guiyi, spliting
from data_making import fft_transformer
from Saving_model_pb import SaveImport_model
from datagenerator import checkclassifier

class Resnet:

    def __init__(self, x, filters, kernel_size, name, padding='same', activation=tf.nn.relu,
                 kernel_initializer=tf.keras.initializers.TruncatedNormal):
        '''
        残差类属性初始化函数
        :param x: 待输入张量, Tensor/Variable
        :param filters: 卷积核个数, int
        :param kernel_size: 卷积核长宽尺寸, list
        :param name: 节点名, str
        :param padding: 标记是否自动补零, str
        :param activation: 激活函数, func
        :param kernel_initializer: 参数初始化函数, func
        '''
        self.__x = x
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__padding = padding
        self.__activation = activation
        self.__kernel_initializer = kernel_initializer
        self.__name = name

    def resnet_2layers(self):
        '''
        两层卷积的子网络结构
        :return: 子网络残差结构输出
        '''
        conv1 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       activation=self.__activation,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        conv2 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(conv1)
        if self.__x.get_shape().as_list()[-1] != conv2.get_shape().as_list()[-1]:
            x = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        else:
            x = self.__x
        combination = tf.keras.layers.Add()([conv2, x])
        relu = tf.keras.layers.ReLU(name=self.__name)(combination)
        return relu

    def resnet_3layers(self):
        conv1 = tf.keras.layers.Conv2D(filters=self.__filters // 4,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       activation=self.__activation,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        conv2 = tf.keras.layers.Conv2D(filters=self.__filters // 4,
                                       kernel_size=self.__kernel_size,
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(conv2)
        if self.__x.get_shape().as_list()[-1] != conv2.get_shape().as_list()[-1]:
            x = tf.keras.layers.Conv2D(filters=self.__filters,
                                       kernel_size=[1, 1],
                                       padding=self.__padding,
                                       kernel_initializer=self.__kernel_initializer)(self.__x)
        else:
            x = self.__x
        combination = tf.keras.layers.Add()([conv3, x])
        relu = tf.keras.layers.ReLU(name=self.__name)(combination)
        return relu

def cnnlstm_regression(x_f, x_l, is_training):
    '''
    神经网络层
    :param x_f: 4个与密度无关特征
    :param x_l: 100个密度特征
    :param is_training: 指示是否在训练
    :return: 神经网络最后输出
    '''
    with tf.name_scope('cnn'):
        x_conv = tf.reshape(tensor=x_l, shape=[-1, 10, 10, 1], name='x_conv')
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv1')(x_conv)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')(conv1)
        bn1 = tf.keras.layers.BatchNormalization(name='bn_input1')
        bn_input1 = bn1(inputs=pool1, training=is_training)
        # 添加bn层节点依赖
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn1.updates)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv2')(bn_input1)
        # conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
        #                                kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv3')(conv2)
        pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool2')(conv2)
        # bn2 = tf.keras.layers.BatchNormalization(name='bn_input2')
        # bn_input2 = bn2(inputs=pool2, training=is_training)
        # # 添加bn层节点依赖
        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn2.updates)
        # conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[2, 2], padding='same', activation=tf.nn.relu,
        #                                kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv4')(bn_input2)
        flat1 = tf.keras.layers.Flatten(name='flat1')(pool2)
    with tf.name_scope('rnn'):
        x_lstm = tf.reshape(tensor=flat1, shape=[-1, 24, 24], name='x_lstm')
        # lstm1 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=True, name='lstm1')(x_lstm)
        lstm2 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=False, name='lstm1')(x_lstm)
        flat2 = tf.keras.layers.Flatten(name='flat2')(lstm2)
    with tf.name_scope('dnn'):
        x_dnn = tf.concat(values=[flat2, x_f], axis=1)
        x_fc1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')(x_dnn)
        x_dpt1 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt1')(inputs=x_fc1, training=is_training)
        x_fc2 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')(x_dpt1)
        x_dpt2 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt2')(inputs=x_fc2, training=is_training)
        output = tf.keras.layers.Dense(units=1, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(x_dpt2)
    return output

# def res_regression(x_f, x_l, is_training):
#     '''
#     残差回归网络层
#     :param x_f: 4个与密度无关特征
#     :param x_l: 100个密度特征
#     :param is_training: 指示是否在训练
#     :return: 神经网络最后输出
#     '''
#     with tf.name_scope('sub_cnn'):
#         x_reshape = tf.reshape(tensor=x_l, shape=[-1, 10, 10, 1], name='x_reshape')
#         conv = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu,
#                                        kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv1')(x_reshape)
#         pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same', name='pool1')(conv)
#         resnet = Resnet(x=pool1, filters=64, kernel_size=[3, 3], name='resnet1')
#         res1 = resnet.resnet_2layers()
#
#         resnet2 = Resnet(x=res1, filters=128, kernel_size=[3, 3], name='resnet2')
#         res2 = resnet2.resnet_2layers()
#
#         resnet3 = Resnet(x=res2, filters=256, kernel_size=[3, 3], name='resnet3')
#         res3 = resnet3.resnet_2layers()
#
#         resnet4 = Resnet(x=res3, filters=256, kernel_size=[3, 3], name='resnet4')
#         res4 = resnet4.resnet_2layers()
#
#         # resnet5 = Resnet(x=res4, filters=128, kernel_size=[3, 3], name='resnet5')
#         # res5 = resnet5.resnet_2layers()
#
#         flat = tf.keras.layers.Flatten(name='flat')(res4)
#     with tf.name_scope('sub_dnn'):
#         x_dnn = tf.concat(values=[flat, x_f], axis=1)
#         x_fc1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
#                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
#                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')(x_dnn)
#         x_dpt1 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt1')(inputs=x_fc1, training=is_training)
#         x_fc2 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
#                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
#                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')(x_dpt1)
#         x_dpt2 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt2')(inputs=x_fc2, training=is_training)
#         output = tf.keras.layers.Dense(units=1, activation=tf.nn.relu, use_bias=True,
#                                        kernel_initializer=tf.keras.initializers.TruncatedNormal,
#                                        bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(x_dpt2)
#         output2 = tf.keras.activations.softmax(x=output)
#         return output2

def acc_regression(Threshold, y_true, y_pred):
    '''
    回归精确度（预测值与实际值残差在阈值范围内的数量/样本总数）
    :param Threshold: 预测值与实际值之间的绝对值之差阈值
    :param y_true: 样本实际标签
    :param y_pred: 样本预测结果
    :return: 精确率，返回计算图节点op，结果需要放在计算图中运行转为ndarray
    '''
    # 残差布尔向量
    is_true = tf.abs(y_pred - y_true) <= Threshold
    is_true_cast = tf.cast(is_true, tf.float32)
    acc_rate_regression = tf.reduce_mean(is_true_cast)
    return acc_rate_regression


def session(dataset_path, train_path='', test_path=''):
    '''
    节点连接
    :param dataset_path: 数据集路径
    :param train_path: 训练集数据路径,默认为空
    :param test_path: 测试集数据路径,默认为空
    :return: None
    '''
    #导入数据集
    dataset = LoadFile(p=dataset_path)
    g1 = tf.Graph()
    with g1.as_default():
        with tf.name_scope('placeholder'):
            x_f = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='x_f')
            x_l = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='x_l')
            y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
            learning_rate = tf.placeholder(dtype=tf.float32, name='lr')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        # output = res_regression(x_f=x_f, x_l=x_l, is_training=is_training)
        output = cnnlstm_regression(x_f=x_f, x_l=x_l, is_training=is_training)
        with tf.name_scope('prediction'):
            loss = tf.reduce_mean(tf.square(output - y))
            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            # opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            acc1 = acc_regression(Threshold=0.1, y_true=y, y_pred=output)
            acc2 = acc_regression(Threshold=0.16, y_true=y, y_pred=output)
            acc3 = acc_regression(Threshold=0.2, y_true=y, y_pred=output)
            acc4 = acc_regression(Threshold=0.3, y_true=y, y_pred=output)
        with tf.name_scope('etc'):
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g1) as sess:
        sess.run(init)
        # 划分训练集和测试集
        train_data, test_data = spliting(dataset, 3369)
        for i in range(60000): #20000
            for data in input(dataset=train_data, batch_size=1000):
                _ = sess.run(opt, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-1], y: data[:, -1][:, np.newaxis],
                                             learning_rate: 1e-2, is_training: False})
                if i % 100 == 0:
                    loss_ = sess.run(loss, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-1], y: data[:, -1][:, np.newaxis],
                                                      is_training: False})
                    acc_1 = sess.run(acc1, feed_dict={x_f: data[:, :4], x_l: data[:, 4:-1], y: data[:, -1][:, np.newaxis],
                                                     is_training: False})
                    acc_2 = sess.run(acc2,
                                     feed_dict={x_f: data[:, :4], x_l: data[:, 4:-1], y: data[:, -1][:, np.newaxis],
                                                is_training: False})
                    acc_3 = sess.run(acc3,
                                     feed_dict={x_f: data[:, :4], x_l: data[:, 4:-1], y: data[:, -1][:, np.newaxis],
                                                is_training: False})
                    acc_4 = sess.run(acc4,
                                     feed_dict={x_f: data[:, :4], x_l: data[:, 4:-1], y: data[:, -1][:, np.newaxis],
                                                is_training: False})
            if i % 100 == 0:
                acc_5 = sess.run(acc1, feed_dict={x_f: test_data[:, :4], x_l: test_data[:, 4:-1],
                                                 y: test_data[:, -1][:, np.newaxis], is_training: False})
                acc_6 = sess.run(acc2, feed_dict={x_f: test_data[:, :4], x_l: test_data[:, 4:-1],
                                                 y: test_data[:, -1][:, np.newaxis], is_training: False})
                acc_7 = sess.run(acc3, feed_dict={x_f: test_data[:, :4], x_l: test_data[:, 4:-1],
                                                 y: test_data[:, -1][:, np.newaxis], is_training: False})
                acc_8 = sess.run(acc4, feed_dict={x_f: test_data[:, :4], x_l: test_data[:, 4:-1],
                                                 y: test_data[:, -1][:, np.newaxis], is_training: False})
                print('第%s轮训练集损失函数值为: %s  训练集准确率为: %s:%s %s:%s %s:%s %s:%s  测试集准确率为: %s:%s %s:%s %s:%s %s:%s' %
                      (i, loss_, 0.1, acc_1, 0.16, acc_2, 0.2, acc_3, 0.3, acc_4, 0.1, acc_5, 0.16, acc_6, 0.2, acc_7, 0.3, acc_8))
        tf.summary.FileWriter('log/regression_cnnlstm', sess.graph)
        # 保存模型到文件当前脚本文件路径下的pb格式
        saving_model = SaveImport_model(sess_ori=sess, file_suffix=r'/cnnlstm',
                                        ops=(output, x_f, x_l, is_training), usefulplaceholder_count=4,
                                        pb_file_path=r'/home/xiaosong/桌面/regression_cnnlstm')
        saving_model.save_pb()

if __name__ == '__main__':
    # p = r'/home/xiaosong/桌面/oldenburg相关数据/data_oldenburg/OLDENBURG_all.pickle'
    # dataset = LoadFile(p)
    # # checkclassifier(dataset[:, -1])
    # # 去除半径为0.01的数据
    # dataset_pd = pd.DataFrame(data=dataset, columns=[str(i) for i in range(dataset.shape[-1])])
    # print(dataset_pd.values.shape)
    # # 筛选出经过初分类器后类别标记非0.01的所有数据
    # dataset_pd = dataset_pd.loc[dataset_pd['%s' % (dataset.shape[-1]-1)] != 0.01]
    # dataset = dataset_pd.values
    # print(dataset.shape)
    # dataset_4feature, dataset_dense, label = dataset[:, :4], dataset[:, 4:-1], dataset[:, -1][:, np.newaxis]
    # dataset_fft = fft_transformer(dataset_dense, 100)
    # dataset = np.hstack((dataset_4feature, dataset_fft, label))
    # dataset_guiyi = guiyi(dataset)
    # print(dataset_guiyi.shape)
    # label = dataset_guiyi[:, -1]
    # label = (label - np.min(label)) / (np.max(label) - np.min(label))
    # dataset_guiyi = np.hstack((dataset_guiyi[:, :-1], label[:, np.newaxis]))
    # print(np.max(dataset_guiyi, axis=0))
    # SaveFile(data=dataset_guiyi, savepickle_p=r'/home/xiaosong/桌面/OLDENBURG_fftguiyi.pickle')

    p = r'/home/xiaosong/桌面/OLDENBURG_fftguiyi.pickle'
    session(dataset_path=p)
#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: classifier_first
@time: 2019/8/27 下午5:40
'''
import tensorflow as tf
def layers(x_f, x_l, is_training):
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
        bn2 = tf.keras.layers.BatchNormalization(name='bn_input2')
        bn_input2 = bn2(inputs=pool2, training=is_training)
        # 添加bn层节点依赖
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn2.updates)
        # conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[2, 2], padding='same', activation=tf.nn.relu,
        #                                kernel_initializer=tf.keras.initializers.TruncatedNormal, name='conv4')(bn_input2)
        flat1 = tf.keras.layers.Flatten(name='flat1')(bn_input2)
    with tf.name_scope('rnn'):
        x_lstm = tf.reshape(tensor=flat1, shape=[-1, 24, 24], name='x_lstm')
        # lstm1 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=True, name='lstm1')(x_lstm)
        lstm2 = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=False, name='lstm2')(x_lstm)
        flat2 = tf.keras.layers.Flatten(name='flat2')(lstm2)
    with tf.name_scope('dnn'):
        x_dnn = tf.concat(values=[flat2, x_f], axis=1)
        x_fc1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, use_bias=True,
                                      kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                      bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc1')(x_dnn)
        x_dpt1 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt1')(inputs=x_fc1, training=is_training)
        # x_fc2 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu, use_bias=True,
        #                               kernel_initializer=tf.keras.initializers.TruncatedNormal,
        #                               bias_initializer=tf.keras.initializers.TruncatedNormal, name='x_fc2')(x_dpt1)
        # x_dpt2 = tf.keras.layers.Dropout(rate=0.2, name='x_dpt2')(inputs=x_fc2, training=is_training)
        output = tf.keras.layers.Dense(units=3, activation=tf.nn.relu, use_bias=True,
                                       kernel_initializer=tf.keras.initializers.TruncatedNormal,
                                       bias_initializer=tf.keras.initializers.TruncatedNormal, name='output')(x_dpt1)
        output = tf.keras.activations.softmax(x=output)
    return output

if __name__ == '__main__':
    pass

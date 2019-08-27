#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: Saving_model_pb
@time: 2019/8/27 下午5:42
'''
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
class SaveImport_model:
    '''
    将模型写入序列化pb文件
    '''
    def __init__(self, sess_ori, file_suffix, ops, usefulplaceholder_count, pb_file_path=os.getcwd()):
        '''
        构造函数
        :param sess_ori: 原始会话实例对象(sess)
        :param file_suffix: type= str, 存储模型的文件名后缀
        :param ops: iterable, 节点序列（含初始输入节点x）
        :param usefulplaceholder_count: int, 待输入有用节点(placeholder)数量, 默认为ops列表长度
        :param pb_file_path: str, 获取pb文件保存路径前缀
        '''
        self.__sess_ori = sess_ori
        self.__pb_file_path = pb_file_path
        self.__file_suffix = file_suffix
        self.__ops = ops
        self.__usefulplaceholder_count = usefulplaceholder_count

    def save_pb(self):
        '''
        保存计算图至指定文件夹目录下
        :return: None
        '''
        # 存储计算图为pb格式,将所有保存后的结点名打印供导入模型使用
        #设置output_node_names列表(含初始输入x节点)
        output_node_names = ['{op_name}'.format(op_name = per_op.op.name) for per_op in self.__ops]
        # Replaces all the variables in a graph with constants of the same values
        constant_graph = graph_util.convert_variables_to_constants(self.__sess_ori,
                                                                   self.__sess_ori.graph_def,
                                                                   output_node_names= output_node_names[:-self.__usefulplaceholder_count])
        # 写入序列化的pb文件
        with tf.gfile.FastGFile(self.__pb_file_path + '/' + 'model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # Builds the SavedModel protocol buffer and saves variables and assets
        # 在和project相同层级目录下产生带有savemodel名称的文件夹
        builder = tf.saved_model.builder.SavedModelBuilder(self.__pb_file_path + self.__file_suffix)
        # Adds the current meta graph to the SavedModel and saves variables
        # 第二个参数为字符列表形式的tags – The set of tags with which to save the meta graph
        builder.add_meta_graph_and_variables(self.__sess_ori, ['cpu_server_1'])
        # Writes a SavedModel protocol buffer to disk
        # 此处p值为生成的文件夹路径
        p = builder.save()
        print('计算图保存路径为: ', p)
        for i in output_node_names:
            print('节点名称为:' + i)

def use_pb(sess_new, pb_file_path, file_suffix):
    '''
    将计算图从指定文件夹导入至工程
    :param sess_new: 待导入节点的新会话对象
    :return: None
    '''
    # Loads the model from a SavedModel as specified by tags
    tf.saved_model.loader.load(sess_new, ['cpu_server_1'], pb_file_path + file_suffix)

def import_ops(sess_new, op_name):
    '''
    获取图中的某一个计算节点
    :param sess_new: 待带入节点的新会话对象
    :param op_name: 计算节点名
    :return: 计算节点
    '''
    op = sess_new.graph.get_tensor_by_name('%s:0' % op_name)
    return op

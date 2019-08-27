#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: test
@time: 2019/8/27 下午5:14
'''
a = {2:3, 3:4, 1:5}
print(sorted(a.items(), key=lambda i: i[0]))
b = a.items()
print(list(b))
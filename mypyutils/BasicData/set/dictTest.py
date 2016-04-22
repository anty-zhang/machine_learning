# -*- coding:utf-8 -*-
__author__ = 'anty'

birds = {'eagle' : 999 , 'snow goose' : 33}

#判断某个键在字典中，用k in dict
if 'eagle' in birds:
    print('eagle have been seen')


#从字典中移除某个条件时可使用    del dict[k]
#基本方法：clear   get keys  items values  update
#两种迭代方法：
#   for(key,value) in dict.items():
#   for (key,value) in dict.iteritems():
#get常用法：birds[key]  = birds.get(key,0) + 1


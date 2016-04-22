# -*- coding=utf-8 -*-

#关联分析：从大规模数据集中寻找物品间的隐含信息。主要目标是：①发现频繁项集②发现关联规则
#频繁项集合：经常出现在一起的物品的集合
#关联   规则：暗示两种物品之间可能存在很强的关系

#支持度（support）：一个项集的支持度定义为数据集中包含该项集的记录所占的比例。即包含该项的集合的个数/总的集合个数
#可信度（cofidence）：是针对一条诸如{尿布} --->{啤酒}的关联规则来定义的。
#这条规则的可信度可定义为：（3/5）支持度{尿布，啤酒}/(（4/5）支持度{尿布} = 3/4，这意味着在包含尿布的多有记录中，
#关联规则对其中的3/4的记录都是适用的。

#apriori原理：可以减少可能感兴趣的项集。apriori原理是说，如果某个项集是频繁的，那么它的所有子集也是频繁的。
#反过来说，如果一个项集是非频繁集，那么它的所有超集也是非频繁集。


#问题1：为什么关联规则中，如果项集中有三个元素，为什么只计算1个 -> 2个
#而不计算2个  --->1个？？？？？？？？？？？？？？？

import apriori
from votesmart import votesmart

dataSet = apriori.loadDataSet()
#C1 = apriori.createC1(dataSet)
#print ("C1 is %s"  % C1)
#D = map(set,dataSet)
#print ( "%r"  % D)
#L1,suppData0 = apriori.scanD(list(D), list(C1), 0.5)
#print (L1)
#print (suppData0)

L,suppData = apriori.apriori(dataSet, 0.5)
print ("L is" , L)
print ("suppData is" , suppData)
#L is [[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})], [frozenset({3, 5}), frozenset({1, 3}), frozenset({2, 5}), frozenset({2, 3})], [frozenset({2, 3, 5})], []]
#suppData is {frozenset({5}): 0.75, frozenset({3}): 0.75, frozenset({2, 3, 5}): 0.5, frozenset({3, 5}): 0.5, frozenset({2, 3}): 0.5, frozenset({2, 5}): 0.75, frozenset({1}): 0.5, frozenset({1, 3}): 0.5, frozenset({2}): 0.75}

#关联规则挖掘
rules = apriori.generateRules(L, suppData, 0.7)

print ("rules is " ,rules)

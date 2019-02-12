# -*- coding:utf-8 -*-
import fpGrowth

#算法基本过程：
#1.创建FP树的数据结构
#2.第一次遍历数据集会获得每个元素项的出现频率。 去掉不满足支持度的元素项
#3.对每个事务（即每个记录）中的集合进行排序。排序基于元素项的绝对出现频率来进行
#4.构建FP树。从空集开始，向其中不断添加频繁项集。即在构建时，读入每个事务中的项集，并将其添加到已存在的路径中。
#    如果树中已经存在现有元素，则增加现有元素的值
#    如果该路径不存在，则创建一条新路径。

###测试FP数的数据结构
#rootNode = fpGrowth.treeNode('pyramid',9,None)
#rootNode.children['eye'] = fpGrowth.treeNode('eye',13,None)
#rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix',3,None)
#rootNode.disp()


simData = fpGrowth.loadSimpDat()
initSet = fpGrowth.createInitSet(simData)
myFpTree,myHeaderTab = fpGrowth.createTree(initSet, 3)
myFpTree.disp()

myCondPats = fpGrowth.findPrefixPath('r', myHeaderTab['r'][1])
print ('myCondPats is' , myCondPats)

freqItems = []
fpGrowth.mineTree(myFpTree, myHeaderTab, 3, set ([]), freqItems)
print('频繁项集 is' , freqItems)


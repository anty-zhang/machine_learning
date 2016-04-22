# -*- coding:utf-8 -*-
import trees

myData,myLabels = trees.createDataSet()
testLabels = myLabels.copy()
print ('myData is ' , myData)

#计算无序数据集的香农熵
#myShannonEnt = trees.calcShannonEnt(myData)
#print ('myShannonEnt is ' , myShannonEnt )

###测试划分数据集函数
#mySplitDat = trees.splitDataSet(myData, 1, 0)
#print ('mySplitDat is ' , mySplitDat )

#myBestData = trees.chooseBestFeatureToSplit(myData)
#print ('myBestData is ' , myBestData )

myTree = trees.createTree(myData, myLabels)
print ('myTree is ' ,myTree)


#测试训练集
print ('testLabels is ' ,testLabels)
testResult = trees.classify(myTree, testLabels, [1,1])
print ('testResult is ' ,testResult)

#trees.storeTree(myTree, 'classifierStorage.txt')
fromFileTree = trees.grabTree('classifierStorage.txt')
print ('fromFileTree is' , fromFileTree)
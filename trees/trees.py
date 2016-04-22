# -*- coding:utf-8 -*-

'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

#function：计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    
    #计算数据集中每个标签类别的个数，以key：value的形式存放在labelCounts
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries       #选择该分类的概率
        shannonEnt -= prob * log(prob,2) #log base 2      #log(prob,2)：量化的信息计算
    return shannonEnt

#function：按着给定特征值划分数据集
#input param：
#    dataSet：原始数据集
#    axis：划分数据集的特征
#    value：划分数据集的特征的特征值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #抽取划分信息
            #用extend和append两个函数，将特征值axis去掉，保留该记录剩余的剩余信息
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#function：选择最好的划分数据集的方式
#input param：
#    dataSet：（1）数据必须是由列表元素组成的列表，而且列表元素都必须具有相同的长度
#                （2）实例的最后一列是当前实例的类别标签
#output：
#    bestFeature：返回最佳特征值的索引
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    
    #计算了整个数据集的原始香农熵
    #用于与划分之后的数据集计算的熵值进行比较
    baseEntropy = calcShannonEnt(dataSet)
    
    
    bestInfoGain = 0.0;     #保存临时香农熵
    bestFeature = -1        #保存最佳特征值的下标
    
    #遍历数据集中的所有特征值的索引
    for i in range(numFeatures):        #iterate over all the features
        #创建唯一的分类列标
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        
        newEntropy = 0.0         #存放按每种特征值划分方式的香农熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)      #对每个特征中的，每个特征值划分一次数据集
            prob = len(subDataSet)/float(len(dataSet))        #计算某个特征值占整个数据集的比率
            newEntropy += prob * calcShannonEnt(subDataSet)     #将split的数据集的香农熵 * 该split数据集占整个数据集的比率
                                                                                                #然后相加所有的特征
        #计算特征值中的信息增益，返回最好特征划分的索引值
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

#function：当满足第二个停止条件：使用完了所有特征，仍不能将数据集划分成仅包含唯一类别的分组
#    则挑选出现次数最多的类别作为返回值
#input param：
#    classList：分类名称的列表
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#function：创建树
#input param：
#    dataSet：数据集
#    labels：标签列表
def createTree(dataSet,labels):
    #classList存放了所有数据集的类别标签
    classList = [example[-1] for example in dataSet]
    
    #第一个停止条件：所有类标签完全相同，则直接返回该类标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    
    #第二个停止条件：使用完了所有特征，仍不能将数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    
    
    bestFeat = chooseBestFeatureToSplit(dataSet)     #bestFeat：当前数据集选取最好的特征索引
    bestFeatLabel = labels[bestFeat]        #特征类别标签
    
    #实例myTree is  {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    myTree = {bestFeatLabel:{}}    #字典变量myTree存储了树的所有信息
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  #得到列表包含的所有属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

#function：决策树分类函数
#input param：
#    inputTree：决策树
#    featLabels：用于训练决策树的特征值标签
#    testVec：测试向量，必须和训练决策树的向量长度一样，不包含类别标签即可
#output：classLabel返回叶子结果对应的分类
def classify(inputTree,featLabels,testVec):
    #firstStr = inputTree.keys()[0]
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)    #找到特征值索引
    key = testVec[featIndex]                      #取出测试向量中指定索引的value值
    valueOfFeat = secondDict[key]            #该value值对应的结果
    
    #如果valueOfFeat对应的是叶子节点，则返回分类标签classLabel
    #如果是一个子树，那么则需要递归调用classify
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

#function：在硬盘上存储决策树
def storeTree(inputTree,filename):
    import pickle        #序列化对象
    #fw = open(filename,'w')
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

#function：在硬盘上获取决策树
def grabTree(filename):
    import pickle
    #fr = open(filename)
    fr = open(filename,'rb')
    return pickle.load(fr)




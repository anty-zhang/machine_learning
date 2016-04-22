# -*- coding=utf-8 -*-

'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
   # return [[1, 3, 4], [1,2, 3, 5], [1, 2, 3, 5], [2, 5]]


#构建第一个项集列表C1
#C1是大小为1的所有项集的集合
#C1中不是简单的添加一个物品项，而是添加包含该物品项的一个列表，目的是为每个物品项构建一个集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if  not [item] in C1:
                C1.append([item])
                
    C1.sort()
    #frozenset 表示数据集C1是不可改变的
    return map(frozenset, C1)#use frozen set so we
                            #can use it as a key in a dict    

#param desc:
#D：数据集；Ck：候选项集列表；minSupport：感兴趣项集的最小支持度
#func：负责从C1生成L1
#注意：这里的D和CK需要将map转为list来操作，否则会出现错误
def scanD(D, Ck, minSupport):
    #遍历数据集中所有交易记录和C1中的所有候选集
    #如果C1中的集合是记录的一部分，那么增加字典中的计数值
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):        #如果can是tid的子集，则返回true
                #if not ssCnt.has_key(can): ssCnt[can]=1
                if not can in ssCnt.keys(): 
                    ssCnt[can] = 1
                else: ssCnt[can] += 1


    #计算支持度，不满足支持度要求的集合不会输出
    numItems = float(len(D))   #计算输入数据集的项数
    print("numItems is %d " % numItems)
    retList = []              #字典元素
    supportData = {}    #频繁项的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)                   #保存大于支持度的字典元素
            supportData[key] = support     #保存大于支持度的支持度值
    return retList, supportData


#input param：LK：频繁项集列表
#K：项集元素个数
#out param：CK
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; 
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList


def apriori(dataSet, minSupport = 0.5):
    #C1 = createC1(dataSet)
    #D = map(set, dataSet)
    C1 = list(createC1(dataSet))     #生成候选项集列表
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)   #从C1生成L1
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


#function：关联规则生成函数
#input param：
#L：频繁项集列表
#supportData：包含那些频繁项集支持数据的字典
#minConf：可信度阀值
#output：生成可信度的规则列表
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    
    #由于无法从单个元素项集中构建关联规则，因为i从1开始
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            #遍历每个频繁项集，并对每个频繁项集创建只包含单个元素集合的列表H1
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         


#function：对规则进行评估
#input param：
#freqSet：单个频繁项集
#H：单个频繁项集中的单个元素集合
#supportData：包含那些频繁项集支持数据的字典
#brl：可信度规则列表
#minConf：最小可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        
        #注意这种计算可信度的方法
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        print (freqSet-conseq,'-->',conseq,'conf:',conf)
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


#function：生成候选集规则
#input param：
#freqSet：单个频繁项集
#H：单个频繁项集中的单个元素集合
#supportData：包含那些频繁项集支持数据的字典
#brl：可信度规则列表
#minConf：最小可信度
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])    #首先计算H中的频繁集大小m
    if (len(freqSet) > (m + 1)): #try further merging     ###这里有疑问？？？？？？
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print (itemMeaning[item])
        print ("           -------->")
        for item in ruleTup[1]:
            print (itemMeaning[item])
        print ("confidence: %f" % ruleTup[2])
        print  ()     #print a blank line
        

import urllib.request as urllib2
from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        #try:
        billDetail = votesmart.votes.getBill(billNum) #api call
        print ('bill Detail is' , billDetail)
        for action in billDetail.actions:
            if action.level == 'House' and \
            (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                actionId = int(action.actionId)
                print ('bill: %d has actionId: %d' % (billNum, actionId))
                actionIdList.append(actionId)
                billTitleList.append(line.strip().split('\t')[1])
        #except:
        #   print ("problem getting bill %d" % billNum)
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList
        
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print ('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print ("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning




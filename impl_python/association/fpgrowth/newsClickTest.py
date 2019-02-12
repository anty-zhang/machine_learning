# -*- coding:utf-8 -*-
import fpGrowth

parseData = [line.split() for line in open('kosarak.dat').readlines()]
#print(parseData)
initSet = fpGrowth.createInitSet(parseData)
myFPTree,myHeaderTab = fpGrowth.createTree(initSet, 100000)
myFreqList = []
fpGrowth.mineTree(myFPTree, myHeaderTab, 100000, set([]), myFreqList)
print ('myFreqList is ' , myFreqList)
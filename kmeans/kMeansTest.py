# -*- coding:utf-8 -*-

import kMeans
from numpy import  *

datMat = mat(kMeans.loadDataSet("testSet.txt"))

'''
myCentroids,clusterAssing = kMeans.kMeans(datMat, 4)
print("myCentroids is %s " % myCentroids)
print("clusterAssing is %s " % clusterAssing)
'''


#kMeans test example two
dataMat2 = mat(kMeans.loadDataSet('testSet2.txt'))
centList,myNewAssment = kMeans.biKmeans(dataMat2, 3)
print(centList)

#geoResult = kMeans.geoGrab('1 VA Center', 'Augusta,ME')



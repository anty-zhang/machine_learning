# -*- coding:utf-8 -*-
import pca
from numpy import *

dataMat = pca.loadDataSet('testSet.txt')
#print (dataMat)
#dataMat = mat([[1,2],[3,4]])
lowDMat,reconMat = pca.pca(dataMat, 1)
m = shape(lowDMat)
print ('m is ' , m)
print ('lowDMat is' , lowDMat)
print ('reconMat is' , reconMat)


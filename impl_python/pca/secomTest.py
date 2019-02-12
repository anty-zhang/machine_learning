# -*- coding:utf-8 -*-
import pca
from numpy import *

dataMat= pca.replaceNanWithMean()
'''
myLowDDataMat, myReconMat = pca.pca(dataMat, 20)
print ('myLowDDataMat is' , myLowDDataMat)
print ('myReconMat is' , myReconMat)
'''

topNfeat = 20
meanVals = mean(dataMat, axis=0)     #按列求平均值
meanRemoved = dataMat - meanVals #remove mean

covMat = cov(meanRemoved, rowvar=0)     #求协方差矩阵
eigVals,eigVects = linalg.eig(mat(covMat))     #特征值和特征向量
#print ('eigVals is ' ,  eigVals)
eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest  返回的是下标，默认是升序排序
eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
redEigVects = eigVects[:,eigValInd]

print (eigVals.sum())


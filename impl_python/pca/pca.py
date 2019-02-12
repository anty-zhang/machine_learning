# -*- coding:utf-8 -*-

'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

#input param:
#    dataMat：用于PCA操作的数据集
#    topNfeat：应用的N个特征
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)     #按列求平均值
    meanRemoved = dataMat - meanVals #remove mean
    
    covMat = cov(meanRemoved, rowvar=0)     #求协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))     #特征值和特征向量
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest  返回的是下标，默认是升序排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest   #获取列向量
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    #lowDDataMat：降维之后的矩阵
    #reconMat：重构后的矩阵
    return lowDDataMat, reconMat

#function：将NaN替换成平均值
def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat



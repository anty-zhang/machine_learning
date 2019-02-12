# -*- coding:utf-8 -*-
import adaboost
from numpy import *

myData,myLabels = adaboost.loadSimpData()
'''
print ('myData is ' , myData)
print ('myLabels is' , myLabels)

D = mat(ones((5,1))/5)
print ('D is', D)

myBStump,myMError,myBCE = adaboost.buildStump(myData, myLabels, D)
print ('myBStump is', myBStump)
print ('myMError is', myMError)
print ('myBCE is', myBCE)
'''
classiFierArray,classEst = adaboost.adaBoostTrainDS(myData,myLabels,30)
print ('classiFierArray is ',classiFierArray)
aggClassEst = adaboost.adaClassify([[5,5],[0,0]], classiFierArray)
print ('aggClassEst is ' ,  aggClassEst)
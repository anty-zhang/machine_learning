# -*- coding:utf-8 -*-
import adaboost
from numpy import *
import time


time_start = time.time()
datArr,labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
classifierArr,aggBestEst = adaboost.adaBoostTrainDS(datArr, labelArr, 30)

testArr,testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
prediction10 = adaboost.adaClassify(testArr, classifierArr)
m = shape(testArr)[0]
errArr = mat(ones((m,1)))
errCount = errArr[prediction10  != mat(testLabelArr).T].sum()
print ('errCount is ' , errCount , 'error rate is ',errCount/m)
time_end = time.time()
print ("the program spend  %d s" % (time_end - time_start))
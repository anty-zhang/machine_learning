# -*- coding: utf-8 -*-

import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import time



#test knn classify simple start
group,label = kNN.createDataSet()
print ("group is %s,label is %s" % (group,label))
###
result = kNN.classify0([0,0.1], group, label, 3)
print ("the KNN classify result is %s" % result)
#test knn classify simple end


#test dating classify
###datingDatMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
####print("datingDatMat is %s" % datingDatMat)
####show the seconde,third column of the matrix start
###fig = plt.figure();
###ax = fig.add_subplot(111)
####ax.scatter(datingDatMat[:,1],datingDatMat[:,2])  #no color
###ax.scatter(datingDatMat[:,0],datingDatMat[:,1],15.0  *  array(datingLabels),15.0  *  array(datingLabels))  #has color
###plt.show()
#show the seconde,third column of the matrix start


###数值归一化处理
###datingDatMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
###normMat,ranges,minVals = kNN.autoNorm(datingDatMat);


###约会网站测试调用
# kNN.datingClassTest()
#kNN.classifyPersion()

#手写数字识别系统测试条用
'''
time_start = time.time()
kNN.handwritingClassTest()
time_end = time.time()
print ("the program spend  %d s" % (time_end - time_start))
'''

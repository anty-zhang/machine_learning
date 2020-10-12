# -*- coding:utf-8 -*-
import svmMLia
from numpy import *
import time

dataArr,labelArr = svmMLia.loadDataSet("testSet.txt");

'''
time_start = time.time()
b,alphas = svmMLia.smoSimple(dataArr, labelArr, 0.6, 0.001, 40);

print("b is %f\n" % b);
time_end = time.time()
print ("the program spend  %d s" % (time_end - time_start))

for i in range(100):
    if alphas[i] > 0.0:
        print("dataArr[%d] is %s,labelArr[%d] is %f" % (i,dataArr[i],i,labelArr[i]))
''' 


#non kernel version below
'''
time_start = time.time()
b,alphas = svmMLia.smoP(dataArr, labelArr, 0.6, 0.001, 40);
time_end = time.time()
print ("the program spend  %d s" % (time_end - time_start))

for i in range(100):
    if alphas[i] > 0.0:
        print("dataArr[%d] is %s,labelArr[%d] is %f" % (i,dataArr[i],i,labelArr[i]))

ws = svmMLia.calcWs(alphas, dataArr, labelArr);
print ("ws is %s",ws)

dataMat = mat(dataArr);
fx = dataMat[0] * mat(ws) + b
print("the result fx is %f  labelArr[0] is %f" % (fx,labelArr[0]))

fx = dataMat[1] * mat(ws) + b
print("the result fx is %f  labelArr[0] is %f" % (fx,labelArr[1]))

fx = dataMat[2] * mat(ws) + b
print("the result fx is %f  labelArr[0] is %f" % (fx,labelArr[2]))
'''



#kernel version below
#svmMLia.testRbf(0.1);
time_start = time.time()
svmMLia.testDigits(('rbf',10))
time_end = time.time()
print ("the program spend  %d s" % (time_end - time_start))
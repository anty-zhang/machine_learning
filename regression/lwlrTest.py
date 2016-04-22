# -*- coding=utf-8 -*-
import regression
from numpy import *

xArr,yArr = regression.loadDataSet("ex0.txt")

#l0 = regression.lwlr(xArr[0], xArr, yArr, 1.0)
#l1 = regression.lwlr(xArr[0],xArr,yArr,0.001)
#print ("l0 is %s" % l0)
#print ("l1 is %s" % l1)



yHat = regression.lwlrTest(xArr, xArr, yArr, 1.0)
print ("yHat is %s"  % yHat)

xMat = mat(xArr)
#axis=0 按列排序;axis=1 按行排序
#返回xMat下标编号
srtInd = xMat[:,1].argsort(0)
#print("srtInd is %s"  % srtInd)




xSort = xMat[srtInd][:,0,:]    #这是什么意思？？？？？
#print ("xSort is %s"% xSort)



import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],mat(yHat[srtInd]))    #这里有问题？？？？？
ax.scatter(mat(xArr)[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s = 2, c = 'blue')

plt.show()


# -*- coding:utf-8 -*-

import regression
from numpy import *

xArr,yArr = regression.loadDataSet('ex0.txt')
#print ("xArr is %s" % xArr)
ws = regression.standRegres(xArr, yArr)
print ("ws is %s" % ws)

xMat = mat(xArr)
yMat = mat(yArr)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])   #绘制原始图

#绘制模拟图
#xCopy = xMat.copy()
#xCopy.sort(0)
#yHat = xCopy*ws
#fig.plot(xCopy[:,1],yHat)   #这里不能运行，不清楚为什么
#plt.show()


#用相关系数来判断模型的好坏
#即判断预测结果与真实结果之间的拟合程度
corYHat = xMat * ws
mycor = corrcoef(corYHat.T,yMat)    #计算相关系数
print("mycor is %s" % mycor)






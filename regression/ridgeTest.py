# -*- coding:utf-8 -*-
import regression
from numpy import *

abX,abY = regression.loadDataSet("abalone.txt")
ridgeWeight = regression.ridgeTest(abX, abY)
#print ("ridgeWeight is %s"  % ridgeWeight)


#展现回归系数与log（lam）的关系
#lam非常小时，与线性回归一致
#lam非常大时，系数全部缩减成0
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeight)
plt.show()





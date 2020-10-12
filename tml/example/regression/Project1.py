# -*- coding=utf-8 -*-
import regression
from numpy import *

abX,abY = regression.loadDataSet("abalone.txt")
yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1.0)
yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

error01 = regression.rssError(abY[0:99], yHat01)
error1 = regression.rssError(abY[0:99], yHat1)
error10 = regression.rssError(abY[0:99], yHat10)

#结论，使用较小的核可以得到较低的误差
#但较小的核会造成过拟合的，对新数据不定能达到最好的预测效果
print ("error01 is %s"  % error01)     #error01 is 56.7862596807
print ("error1 is %s"  % error1)         #error1 is 429.89056187
print ("error10 is %s"  % error10)     #error10 is 549.118170883

yyHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
yyHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1.0)
yyHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
eerror01 = regression.rssError(abY[100:199], yyHat01)
eerror1 = regression.rssError(abY[100:199], yyHat1)
eerror10 = regression.rssError(abY[100:199], yyHat10)
print ("eerror01 is %s"  % eerror01)     #eerror01 is 33652.8973161
print ("eerror1 is %s"  % eerror1)         #eerror1 is 573.52614419
print ("eerror10 is %s"  % eerror10)     #eerror10 is 517.571190538       #对新数据，k=10得到较好的效果


#和线性做比较
#结论：必须在未知数据上做比较效果才能取到最佳模型
ws = regression.standRegres(abX[0:99], abY[0:99])   #用前100个数据做训练集
yHat = mat(abX[100:199]) * ws
errorLine = regression.rssError(abY[100:199], yHat.T.A)
print ("errorLine is %s"  % errorLine)     #errorLine is 518.636315324



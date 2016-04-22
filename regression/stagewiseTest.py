# -*- coding:utf-8 -*-
import regression
from numpy import *

xArr,yArr = regression.loadDataSet("abalone.txt")
#stageWeight = regression.stageWise(xArr, yArr, 0.01, 200)
#print (stageWeight)

stageWeight = regression.stageWise(xArr, yArr, 0.0001, 50000)
#print (stageWeight)
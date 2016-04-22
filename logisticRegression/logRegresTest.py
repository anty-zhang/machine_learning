# -*- coding=utf-8 -*-
import logRegres
import time

#dataArr,labels = logRegres.loadDataSet()
#logWeight = logRegres.gradAscent(dataArr, labels)

#print ("logWeight is %s" % logWeight)

#logRegres.plotBestFit(logWeight)     #这个画图始终跑不通，以后在调试？？？？？


#实验：从疝气病预测病马的死亡率
#logRegres.colicTest()
time_start = time.time()
logRegres.multiTest()
time_end = time.time()
print ("the program spend  %d s" % (time_end - time_start))
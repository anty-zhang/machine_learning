# -*- coding:utf-8 -*-

'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#function：测试是否有某个值小于或者大于我们正在测试的阀值
#                所有在阀值一边的数据会分到类别-1，而在另一边的数据分到类别+1
#input param：
#    dataMatrix：原始数据集的矩阵
#    dimen：数据集的特征值索引
#    threshVal：正在预测的阀值（即矩阵dimen列的最小值加上j*步长）
#    threshIneq：lt or gt
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0    #注意这种用法
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
#function：在一个加权数据集中循环，并找到最低错误率的单层决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0;    #用于在特征的所有可能值上遍历
    bestStump = {};   #存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m,1)))
    
    #算法
    #（1）将最小错误率minError置为正无穷
    minError = inf #init error sum, to +infinity
    
    #（2）对数据集中每个特征值
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        
        #（3）对每个步长
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            #（4）对每个不等号
            for inequal in ['lt', 'gt']: #go over less than and greater than
                #（5）建立一棵单层决策树，利用加权数据集对它进行测试
                #        如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0   #这实际上是计算分错的label
                weightedError = D.T*errArr  #calc total error multiplied by D
                print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    
    #（6）返回最佳决策树
    #    bestStump：最佳分类器，例如{'dim': 0, 'thresh': 1.3, 'ineq': 'lt'}
    #    minError：最小错误率
    #    bestClasEst：最佳分类结果
    return bestStump,minError,bestClasEst

#function：基于单层决策树的adaboost训练过程
#output：
#    weakClassArr：返回所有弱分类器
#    aggClassEst：所有分类的训练误差和
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []       #保存若分类器
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        #（1）找到最佳单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        
        #（2）计算alpha
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        
        #（3）计算新的权重向量D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        
        #（4）更新累计类别估计值
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        
        #（5）如果错误率等0.0则退出循环
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))    #注意这种计算方式技巧
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

#function：adaboost分类函数
#input param：
#    datToClass：一个或多个待分类样例
#    classifierArr：多个若分类器组成的数组
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)


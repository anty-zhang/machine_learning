# -*- coding: utf-8 -*-

import bayes

'''
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
#print("myVocabList is %s" % myVocabList)
myVocab2Vec = bayes.setOfWords2Vec(myVocabList, listOPosts[0])
print("myVocab2Vec is %s" % myVocab2Vec)
'''

'''
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)    #将文档合并成没有重复的文档
trainMat = []  #存储每个文档的词向量
for postInDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postInDoc))
    
p0v,p1v,pAb = bayes.trainNB0(trainMat, listClasses)

'''

bayes.testingNB()

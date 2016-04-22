# -*- coding:utf-8 -*-
import treePlotter as tp

myTree = tp.retrieveTree(1)
print ('myTree' , myTree)

numLeafs = tp.getNumLeafs(myTree)
depth =  tp.getTreeDepth(myTree)
print ('numLeafs is ' , numLeafs)
print ('depth is ' , numLeafs)

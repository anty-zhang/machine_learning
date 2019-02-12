# -*- coding:utf-8 -*-
import trees
import treePlotter as tp

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
print ('lenses is ' , lenses)
lensesTree = trees.createTree(lenses, lensesLabels)
print ('lensesTree is ' , lensesTree)
tp.createPlot(lensesTree)



[TOC]

# LR做点击率预估优势劣势

## 优势

- 简单

- 并行

## 劣势

- 大量的特征工程

> 连续特征离散化：并对离散化的特征进行one-hot 编码

> 特征组合：为取得非线性特征，需要对特征进行二阶或者三阶的特征组合

- 特征工程存在的问题

> 连续变量切分点如何选择？以及离散化多少份合理？

> 选取哪些特征交叉？多少阶特征(2阶、三阶...)交叉

# gbdt + lr

- 优势: 解决了LR的存在的问题

> 确定切分点不在根据主观经验，而是根据信息增益，客观的选取切分点和份数

> 每棵树从节点到叶子节点的路径，会经过不同的特征，此路径包含了特征组合，而且包含了二阶、三阶甚至更多

- 使用gbdt + lr的原因

> gbdt在线预测比较困难，而且训练复杂度高于LR。因此在实中，可以离线训练gbdt，然后将该模型作为在线ETL的一部分

# 实战

## R

https://github.com/bourneli/data-mining-papers/blob/master/GBDT/gbdt-lr-exp/model_comparation.R


## Python (TODO)

[Feature transformations with ensembles of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)

https://www.cnblogs.com/wkang/p/9657032.html

https://www.jianshu.com/p/96173f2c2fb4

# reference

[Practical Lessons from Predicting Clicks on Ads at Facebook 论文](https://www.semanticscholar.org/paper/Practical-Lessons-from-Predicting-Clicks-on-Ads-at-He-Pan/daf9ed5dc6c6bad5367d7fd8561527da30e9b8dd?p2df)


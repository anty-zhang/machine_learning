

https://blog.csdn.net/china1000/article/details/51106856 (机器学习（四）--- 从gbdt到xgboost)

https://blog.csdn.net/xwd18280820053/article/details/68927422 (关于树的几个ensemble模型的比较（GBDT、xgBoost、lightGBM、RF）)

https://blog.csdn.net/u014248127/article/details/79015803 (RF,GBDT,XGBoost,lightGBM的对比)

http://dogless.farbox.com/post/github/-ji-qi-xue-xi-xgboost_boosted-treeszheng-li (XGBoost:Boosted Trees整理)


# xgboost 面试总结

## RF和gbdt

### 区别

- 随机森林通过减少模型方差提高性能，GBDT减少模型偏差提高性能

- 随机森林可以并行，GDBT只能串行

- 随机森林可以是回归树、分类树，GDBT只能是回归树

- 随机森林不需要进行数据预处理、归一化；GDBT需要进行特征归一化


## xgboost和gbdt不同之处

- xgboost支持线性分类器；传统的GDBT以CART作为分类器

- 节点分类方式不同，xgboost 采用打分函数；GDBT 采用gini系数

- xgboost增加了正则，而gbdt没有正则

- xgboost使用了二阶泰勒展开，而gbdt使用了一阶

- xgboost在特征力度上并行化

## xgboost的主要特性

- 行抽样

- 列抽样

- 特征力度上并行化

- 自定义损失函数



[xgbosot 训练过程](https://juejin.im/post/5b74e892e51d45664153b080)

[当GridSearch遇上XGBoost 一段代码解决调参问题](https://juejin.im/post/5b7669c4f265da281c1fbf96)

[梯度提升的基本原理](https://juejin.im/post/5a1624d9f265da43310d79d5)

[BAT面试题7和8：xgboost为什么用泰勒展开？是有放回选特征吗？](https://cloud.tencent.com/developer/article/1360822)

[第一周笔记：线性回归](https://zhuanlan.zhihu.com/p/21340974?spm=a2c4e.11153940.blogcont608728.7.46292480uevNmr&refer=mlearn)

[数据挖掘面试准备（1）|常见算法（logistic回归，随机森林，GBDT和xgboost）](https://yq.aliyun.com/articles/608728)

[xgboost的原理没你想像的那么难](https://www.jianshu.com/p/7467e616f227)

[scala-spark版本xgboost包使用](https://blog.csdn.net/u010035907/article/details/53418370)
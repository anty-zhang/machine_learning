
[TOC]

# SVM 的特点和不足

## 特点

- svm是一种有理论基础的小样本学习方法. 大样本的问题主要是在与难以训练

- svm 最终的决策函数主要由少数的支持向量所确定，计算的复杂度取决于支持向量的数目，而不是样本空间的维数

# 对偶问题

## 解决问题

- 对偶问题将将原问题中约束转化为对偶问题中的等式约束

- 方便引入核函数

- 改变了问题的复杂度。

> 由特征向量 w 转化为求比例系数 a，在原始问题下，求解的复杂度与样本的维度有关，即w的维度。在对偶问题下，只与样本的数量有关系。

## 寻找最优值的下界


## 对偶问题


# 方向导数与梯度的关系


# reference

[约束下的最优求解：拉格朗日乘数法和KKT条件](https://blog.csdn.net/yujianmin1990/article/details/48494607)

[约束优化的拉格朗日乘子（KKT）](https://zhuanlan.zhihu.com/p/55532322)

[如何理解拉格朗日乘子法？](https://www.zhihu.com/question/38586401)

[对偶问题的理解](https://blog.csdn.net/johnnyhuang39/article/details/81504697)

[如何通俗地讲解对偶问题？尤其是拉格朗日对偶lagrangian duality？](https://www.zhihu.com/question/58584814/answer/158045114)

[SVM面试级推导](https://www.jianshu.com/p/514569d11fd8)

[如何直观形象的理解方向导数与梯度以及它们之间的关系？](https://www.matongxue.com/madocs/222/)

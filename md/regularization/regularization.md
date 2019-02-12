
# references: https://www.cnblogs.com/hellcat/p/7979711.html (『科学计算』L0、L1与L2范数_理解)
# https://blog.csdn.net/fightsong/article/details/53311582 (关于L0，L1和L2范数的规则化)
# https://blog.csdn.net/zchang81/article/details/70208061 (深度学习——L0、L1及L2范数)

# 规则化参数同时最小化误差

## 规则化的理解

> 让模型简单 

> 降低模型复杂度(过多参数导致模型复杂--稀疏 is ok)

> 使用规则项来约束模型(约束来带学习的模型参数w，也就变相的约束了模型)

### 从先验知识角度理解L1和L2正则化与参数稀疏

```bash

L1 -> Laplace先验
L2 -> 高斯先验

```


## 为什么要规则化

> 解决ill-posed或者防止overfitting 或者从统计角度来说，是找一个减少过拟合的估计方法

```bash
overfitting两种情况
1. 样本数量m < 特征维度n
（1）减少特征数量
    人工选择
    L1自动选择
    
（2）增加样本量

2. 拟合函数的系数过大
    L2
```

## L2 范数

> 好处

```bash
1. 学习理论角度 -- 防止过拟合，提升模型的泛化能力
2. 从优化的角度 -- 有助于处理矩阵condition number不好的情况下矩阵求逆困难的问题
```

## L1 和 L2 详细过程

### example1

### example2


# 建设模型是线性回归，不小心多增加了一列重复特征，使用L1和L2优化有什么不同？
> https://stats.stackexchange.com/questions/241471/if-multi-collinearity-is-high-would-lasso-coefficients-shrink-to-0
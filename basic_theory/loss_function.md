[TOC]

# 假设函数

# 损失函数

损失函数是用来评估模型的预测值f(x)和真实值Y不一致的程度，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。

损失函数分为经验风险损失函数和结构风险损失函数。经验风险损失函数指的是 预测结果和实际结果的差别; 结构风险损失函数指的是经验风险损失函数加上正则项。

通式为: $\Theta = arg min \sum_{i=1}^n L(Y, f(x_i; \theta)) + \lambda \phi(\Theta)$

## log Loss (对数损失函数, LR)

- LR 损失函数逻辑介绍

> 在逻辑回归的推导中，它假设样本服从伯努利分布（0-1分布）
> 然后求得满足该分布的似然函数，
> 接着取对数求极值等等。
> 而逻辑回归并没有求似然函数的极值，而是把极大化当做是一种思想，进而推导出它的经验风险函数为：最小化负的似然函数（即max F(y, f(x)) —-> min -F(y, f(x)))。从损失函数的视角来看，它就成了log损失函数了。

- 第一种形式: label取值为0或1

$$\begin{array}{l}
z = {\theta ^T}x + b\\
\widehat y = \sigma (z) = \frac{1}{{1 + {e^{ - z}}}}
\end{array}$$

$$p(y|x) = \left\{ \begin{array}{l}
\widehat y & if \quad y = 1\\
1 - \widehat y & if \quad y = 0
\end{array} \right.$$

$$\begin{array}{l}L(y,\widehat y) =  - y\log (\widehat y) - (1 - y)\log (1 - \widehat y)\\
\frac{{\partial L(y,\widehat y)}}{{\partial {\theta _j}}} = \frac{{L(y,\widehat y)}}{{\partial \widehat y}}\frac{{\widehat {\partial y}}}{{\partial z}}\frac{{\partial z}}{{{\theta _j}}}\\
 = ( - \frac{y}{{\widehat y}} + \frac{{1 - y}}{{1 - \widehat y}})\widehat y(1 - \widehat y){x_j}\\
 = (\widehat y - y){x_j}
\end{array}$$

- 第二种形式: 将label和预测函数放在一起，label取值为1或-1

$$\begin{array}{l}
P\{ y = 1|x\}  = \widehat y = \sigma (z) = \frac{1}{{1 + {e^{ - z}}}}\\
P\{ y =  - 1|x\}  = 1 - P\{ y = 1|x\} \\
 = 1 - \frac{1}{{1 + {e^{ - z}}}}\\
 = \frac{1}{{1 + {e^z}}}\\
 = \frac{1}{{1 + {e^z}}} = \sigma ( - z)\\
P\{ y|x\}  = \sigma (y\widehat y)\\
L(y,\widehat y) =  - \log (\sigma (y\widehat y))\\
\frac{{\partial L(y,\widehat y)}}{{\partial \theta_j }} =  - \frac{1}{{\sigma (y\widehat y)}}\sigma (y\widehat y)(1 - \sigma (y\widehat y))y\frac{{\partial \widehat y}}{{\partial \theta_j }}\\
 = (\sigma (y\widehat y) - 1)y x_j
\end{array}$$

## hinge Loss (折页损失函数, SVM)

## exp Loss (指数损失函数, adaBoost)

## cross-entropy Loss (交叉熵损失函数, softmax)

## quadratic Loss (平方误差损失函数, 线性回归)

> 平方损失函数是线性回归在 假设样本是高斯分布的条件下推导出来的

## absolution Loss (绝对值损失函数)

## 0-1 Loss (0-1 损失函数)



# 优化算法

## 梯度下降法

# refercence

[Deep Learning基础--各个损失函数的总结与比较](https://www.cnblogs.com/shixiangwan/p/7953591.html)

[常见的损失函数(loss function)总结](https://zhuanlan.zhihu.com/p/58883095)

# 第3章 线性模型

**线性模型：**

$$ f(x) = \omega ^T + b$$

 

## 3.1 线性回归

**思路**：

- 使均方误差最小化
- 模型求解方法： **最小二乘法**

#### 一元线性回归

对于$f(x) = \omega x + b$:

$$\omega = \frac{\sum_{i=1}^m y_i(x_i - x_{mean})}{\sum_{i=1}^m x^2 - \frac{1}{m}(\sum_{i=1}^m x_i)^2}$$

$$b = \frac{1}{m} \sum_{i=1}^m(y_i - \omega x_i)$$



#### 多元线性回归

令 W = ($\omega$, b) ，X = (x;1) 则：

$$W = (X^TX)^{-1}X^Ty$$

**如果遇到列数超过行数的情况，可以引入正则化项**

**广义线性模型：**

$$y = g^{-1}(\omega ^T x + b)$$

*注： g(.)是联系函数； 可以将默写非线性函数映射转化为线性模型； 例如对数线性回归模型*



## 3.2 对数几率回归

如何使线性模型进行分类问题？

——**寻找一个单调可微函数将分类任务的真实标记y与线性回归模型的预测值联系起来**

**Sigmoid函数：**

$$y = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\omega ^T + b)}}$$

$$ln\frac{y}{1-y} = \omega ^T x + b$$

其中$\frac{y}{1-y}$被称为几率，$ln\frac{y}{1-y}$被称为对数几率

显然有：

$$p(y = 1|x) = \frac{e^{\omega ^T x + b}}{1 +e^{\omega ^T x + b}}$$

$$P(y = 0|x) = \frac{1}{1+e^{\omega ^T x + b}}$$

**代价函数（交叉熵）：**

![1](https://github.com/Pythonboy/Image/blob/master/ML/11.jpg?raw=true)

#### 博客参考

[逻辑回归 by weixin_39910711](https://blog.csdn.net/weixin_39910711/article/details/81607386)



## 3.3 线性判别分析（Linear Discriminant Analysis)

**LDA思想：**

给定训练样例集，设法将其投影在一条直线上，使相同类别样例点的投影点尽可能地近、异类样例的投影点尽可能地远离；在对新样本进行分类时，将其投影在这条直线上，根据投影点的位置来确定新样本的类别



#### 博客参考

[一文详解LDA主题模型](https://segmentfault.com/a/1190000012215533#articleHeader17)

[主题模型——通俗理解与简单应用](https://blog.csdn.net/qq_39422642/article/details/78730662)



## 3.4 多分类问题

- 一对一（OvO) ： 需要训练N(N-1)/2个分类器
- 一对多（OvR） ： 只需要训练N个分类器
- 多对多（McM）
# 第2章 模型评估与选择

## 2.1 经验误差与过拟合

**错误率**  、 **精度**

**训练误差** 、 **泛化误差**

*注： 更倾向于得到泛化误差小的分类器*

**欠拟合 underfitting**  、 **过拟合overfitting**



## 2.2 评估方法

#### 留出法

直接将数据集D划分为两个互斥的集和，一个作为训练集S，一个作为测试集T

#### 交叉验证法

将数据集D划分为k个大小相似的互斥子集，每个子集都尽可能保持数据分布的一致性，即从D中通过分层采样得到

**k-flod cross validation**

特例： **留一法**

#### 自助法

给定包含m个样本的数据集D，对其采样产生数据集D' ： 每次随机从D中挑选一个样本，将其拷贝至D'中，然后放回D中， 重复m次，得到包含m个样本的数据集D' ； 

**初始数据集D中大概有36.8%的样本不会出现在D'中，可以作为测试集使用**



## 2.3 性能度量

**即对模型的评价标准**

### 错误率与精度

**分类任务**中最常用的两种性能度量标准

**错误率：**

$$ E(f ; D) = \frac{1}{m} \sum_{i=1}^m I(f(x_i) \neq y_i)$$

**精度:**

$$acc(f ; D) = 1 - E(f ; D) = \frac{1}{m} \sum_{i=1}^m I(f(x_i) =  y_i)$$

### 查准率、查全率 与 F1

![img](https://img-blog.csdn.net/20150407224050675?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdmVzcGVyMzA1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

 

公式陈列、定义如下：

| True positive(TP)  | eqv. with hit                       |
| ------------------ | ----------------------------------- |
| True negative(TN)  | eqv. with correct rejection         |
| False positive(FP) | eqv. with false alarm, Type I error |
| False negative(FN) | eqv. with miss, Type II error       |

 

| Sensitivity ortrue positive rate(TPR)eqv. with hit rate, recall | TPR = TP/P = TP/(TP + FN)    |
| ------------------------------------------------------------ | ---------------------------- |
| Specificity(SPC)ortrue negative rate(TNR)                    | SPC = TN/N = TN/(FP + TN)    |
| Precision orpositive prediction value(PPV)                   | PPV = TP/(TP + FP)           |
| Negative predictive value(NPV)                               | NPV = TN/(TN + FN)           |
| Fall-out orfalse positive rate(FPR)                          | FPR = FP/N = FP/(FP + TN)    |
| False discovery rate(FDR)                                    | FDR = FP/(FP + TP) = 1 - PPV |
| Miss Rate orFalse Negative Rate(FNR)                         | FNR = FN/P = FN/(FN + TP)    |
| Accuracy(ACC)                                                | ACC = (TP + TN)/(P + N)      |



**查准率： ** $ P = \frac{TP}{TP + FP}$

**查全率：** $R = \frac{TP}{TP + FN}$

**F1：** $F1 = \frac{2 P R}{P + R}$

**$F_{\beta}$ :** $F_{\beta} = \frac{(1+\beta^2) P R}{(\beta^2 P) + R}$

其中$\beta$度量查全率对查准率的相对重要性； 如果 = 1，则退化为F1， 如果 >1 ， 则查全率有更大影响 ， 如果 <1 ,则查准率有更大影响 

**P-R 曲线： 查准率-查全率曲线** ：

![这里写图片描述](https://img-blog.csdn.net/20170113160327591?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFveWFucWk4OTMy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



### ROC 与 AUC

ROC全称为”受试者工作特征“（Receiver Operating Characteristic) 曲线


ROC(receiver operating characteristic)接受者操作特征，其显示的是分类器的**真正率和假正率**之间的关系，如下图所示：

![这里写图片描述](https://img-blog.csdn.net/20170113155954155?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFveWFucWk4OTMy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

ROC曲线有助于比较不同分类器的相对性能，当FPR小于0.36时M1优于M2，而大于0.36是M2较好。 
ROC曲线小猫的面积为AUC(area under curve)，其面积越大则分类的性能越好，理想的分类器auc=1。

$$TPR = \frac{TP}{TP + FN}$$

$$FPR = \frac{FP}{TN + FP}$$

### **如何绘制ROC曲线**

为了绘制ROC曲线，则分类器应该能输出连续的值，比如在**逻辑回归分类器**中，其以概率的形式输出，可以设定阈值大于0.5为正样本，否则为负样本。因此设置不同的阈值就可以得到不同的ROC曲线中的点。 
下面给出具体的实现过程：

![这里写图片描述](https://img-blog.csdn.net/20170113163432528?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdGFveWFucWk4OTMy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



### 代价敏感错误率与代价曲线

![图5 代价矩阵](http://baogege.info/img/learning-from-imbalanced-data/8.png)



#### 博客参考

[不平衡数据下的机器学习方法简介](http://baogege.info/2015/11/16/learning-from-imbalanced-data/)



## 2.4 偏差和方差

$$ E(f ; D) = bias^2(x) + var(x) + \epsilon^2$$

即泛化误差是由偏差、方差与噪声之和

**偏差** 度量了学习算法的期望预测与真实结果的偏离程度

**方差** 度量了同样大小的训练集的变动所导致的学习性能的变化

**噪声** 表达了当前任务上的任何学习算法所能达到的期望泛化误差的下界

![img](https://img-blog.csdn.net/20180712043755524?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYWxpYW5neXVl/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



## 2.5 比较检验

**解决机器学习中性能比较问题**

### 假设检验

### 交叉验证t检验

### McNemar检验

### Friedman 检验与 Nemenyi后续检验



#### 参考博客

[周志华《机器学习》学习笔记及习题探讨（二)](https://zhuanlan.zhihu.com/p/29248751)

[如何计算McNeMar检验](http://www.atyun.com/25532.html)

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





# 第4章 决策树

## 4.1 划分选择

#### 信息熵

假设样本集合D中第k类样本所占的比例为$p_k$，则信息熵定义为:

$$Ent(D) = - \sum_{k=1}^{|D|} p_k log_2 p_k$$

*注： 其中Ent(D)的值越小，则D的纯度越高*

####  信息增益

设$D^v$为第v个分支结点包含了D中所有在属性a上取值为$a^v$的样本，则信息增益为：

$$ Grain(D,a) = Ent(D) - \sum_{i=1}^{V} \frac{|D^v|}{|D|} Ent(D^v)$$

*注： 信息增益越大，则意味着使用属性a来进行划分所获得的纯度提升越大*

#### 增益率

定义：

$$ Grain_ration(D,a) = \frac{Grain(D,a)}{IV(a)}$$

其中：

$$IV(a) = - \sum_{v=1}^{V} \frac{|D^v|}{|D|} log_2 \frac{|D^v|}{|D|}$$

*注： C4.5算法—— 先从候选划分属性中找到信息增益高于平均水平的属性，再从中选择增益率最高的属性*

#### 基尼系数

数据集D的基尼系数：

$$Gini(D) = 1 - \sum_{k=1}^{|\gama|} p_k^2$$

*注：Gini(D)越小，则数据集D的纯度越高*

属性a的基尼系数：

$$ Gini_index(D,a) = \sum_{v=1}^{V} \frac{|D^v|}{|D|} Gini(D^v)$$

*注：选择那个使得划分之后基尼系数最小的属性作为最优划分属性*



## 4.2 剪枝处理

**解决过拟合问题；**

- 预剪枝（自上而下）： 在每个结点进行划分之前，先用验证集对其划分前后的泛化能力进行考察
- 后剪枝（自下而上）：思路同上，欠拟合风险更小，但其训练时间开销也更加大了



# 第5章 神经网络

## 5.1 神经元模型

定义： **神经网络是由具有适应性的简单单元组成的广泛并行互连的网络，它的组织能够模拟生物神经系统对真实世界物品所作出的交互反应**

![这里写图片描述](https://img-blog.csdn.net/20151127092704023)





## 5.2 感知机与多层网络

**感知机** 由两层网络组成；只能解决线性可分的问题

**多层神经网络** ： 输入层神经元仅仅接受输入，隐层和输出层包含功能神经元

![图1 多层前馈神经网络](https://img-blog.csdn.net/20180123140254140) 



## 5.3 全局最小和局部极小

跳出局部极小，接近全局最小的几种策略：

- 使用随机梯度下降法
- 以多组不同参数初始化多个神经网络
- 使用模拟退火技术



## 5.4 其他常见神经网络

### RBF网络

- 单隐层前馈神经网络
- 使用径向基函数作为隐层神经元激活函数

REF网络模型：

$$ f(x) = \sum_{i=1}^{d} \omega_i\rho(x,c_i)$$

$c_i$ 和$\omega_i$分别是第i个隐层神经元所对应的中心和权重

其中高斯径向基函数：

$$\rho(x,c_i) = e^{-\beta_i|x - c_i|^2}$$

**步骤：**

1. 确定神经元中心$c_i$
2. 利用BP算法确定权重参数

![image_1bhb8rkpfasorc61g01ch11m211g.png-84.7kB](http://static.zybuluo.com/jiemojiemo/f260rrta51v700ntrvzf1iqn/image_1bhb8rkpfasorc61g01ch11m211g.png)



###  ART网络（自适应谐振理论网络）

- "胜者通吃"原则
- 该网络由比较层、识别层、识别阈值、重置模块构成



#### 参考博客

[模拟退火算法](https://blog.csdn.net/xianlingmao/article/details/7798647)

[径向基神经网络RBF](https://blog.csdn.net/ACdreamers/article/details/46327761)

[人工神经网络——径向基函数](https://blog.csdn.net/zb1165048017/article/details/49385359)

[RBF神经网络简单介绍与Matlab实现](https://blog.csdn.net/weiwei9363/article/details/72808496)
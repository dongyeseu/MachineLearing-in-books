# 一 统计学习方法概论

## 1. 1 统计学习

**统计学习的方法：**

- 监督学习
- 非监督学习
- 半监督学习
- 强化学习



## 1.2 监督学习

**预测任务：**

- 回归问题 ： 连续变量
- 分类问题 ： 有限个离散变量
- 标注问题 ： 变量序列



## 1.3 统计学习三要素

**方法 = 模型 + 策略 + 算法**

### 模型

**条件概率分布 或 决策函数**

### 策略

**选取最优模型：**

- 损失函数 ：度量模型一次预测的好坏
- 风险函数 ： 度量平均意义下的模型预测的好坏

**损失函数：**

- 0-1 损失函数 ： $L(Y, f(X)) = (1 ,Y not f(X) | 0 , Y is f(X))$ *注： 分类问题*
- 平方损失函数： $L(Y, f(X)) = (Y - f(X) )^2$
- 绝对损失函数： $L(Y, f(X)) = |Y - f(X) |$
- 对数损失函数： $L(Y, P(Y|X)) =  - log P(Y|X) $

*损失函数值越小，模型越好*

对于给定的一个训练数据集，模型f(X)关于其的平均损失称为经验风险：$R_{emp}(f) = \frac{\sum L(y_i , f(x_i))}{N}$



**经验风险最小化与结构风险最小化**

1.  **经验风险最小化** ： $min R_{erm}(f) = min \frac{\sum L(y_i , f(x_i))}{N}$ 

 要求要有足够大的样本容量； 当样品容量很小时，容易产生”过拟合“现象；

2. **结构风险最小化** ： 为了防止过拟合，等价于正则化； 结构风险在经验风险的基础之上加上表示模型复杂度的正则化项或罚项； $R_{srm}(f) = \frac{\sum L(y_i , f(x_i))}{N} + \alpha J(f)$ *模型f越复杂，复杂度J(f)越大*；因此：  $min R_{srm}(f) = min \frac{\sum L(y_i , f(x_i))}{N} + \alpha J(f)$



## 1.4 模型评估与模型选择

### 训练误差与测试误差

**统计学习的目的不仅仅是为了能够使学习到的模型对已知数据，更是为了能够对未知数据都有很好的预测能力**

对于给定的两种学习方法，**测试误差小的**学习方法具有更好的预测能力，是更有效的方法。

### 过拟合与模型选择

**过拟合**是指学习时选择的模型所包含的参数过多，以至于出现这一模型对已知数据预测得很好，但是对于未知数据预测得很差的现象。

模型的最终目的是为了**使测试误差达到最小**



## 1.5正则化与交叉验证

1. **正则化** ： 正则化项一般都是模型复杂度的单调递增函数，模型越复杂，正则化值就越大 ； $$min \frac{\sum L(y_i , f(x_i))}{N} + \alpha J(f)$$

   **example : 回归问题** ： L1正则化 （$\alpha |w|$) & L2正则化（$\frac{\alpha}{2} |w|^2$) 

   Occam's razor : *在所有模型中，能够很好地解释已知数据并且十分简单才是最好的模型*

2. **交叉验证** ： 训练集（training set) & 验证集(validation set) & 测试集（test set）

- 简单交叉验证
- S折交叉验证
- 留一交叉验证（S=N）



## 1.6 泛化能力

#### 泛化误差

**泛化能力** 是指由某一种学习方法得到的模型对未知数据的预测能力。

#### 泛化误差上界

通过比较**泛化误差上界**能够比较两种学习方法的优劣性；

**性质：**

- 它是样本容量的函数，当样品容量增加时，泛化上界趋于0
- 它是假设空间容量的函数，当假设空间容量越大，模型越难学，泛化误差上界也就越大



## 1.7 生成模型与判别模型

1. 生成方法由数据学习联合概率分布 P(X , Y) ,然后求出条件概率分布 P(Y | X) 作为预测的模型 ： $$ P( Y | X) = \frac{P(X,Y)}{P(X)}$$ example : 朴素贝叶斯法 和 隐马尔可夫模型 **对于给定X生成输出Y的生成关系**
2. 判别方法由数据直接学习决策函数 f(X) 或者条件概率分布 P(Y | X) 作为预测的模型； example : K近邻 、 感知机、 决策树、 支持向量机 等 **对于给定的X输出怎样的输出Y**



## 1.8 分类问题

评价分类器性能的指标一般为**分类准确率** *（accuracy）* ： 即为0-1损失函数

#### 精准率（precision) & 召回率（recall）

*注：通常把关注的类为正类，其他类为负类*

- TP ： 将正类预测为正类
- FP ： 将负类预测为正类
- TN ： 将负类预测为负类
- FN： 将正类预测为负类

*（注：第二个字母P/N代表你的预测，第一个字母F/T代表你预测的正确性）*

**精准率**

$$precision = \frac{TP}{TP + FP}$$

**召回率**

$$recall = \frac{TP}{TP + FN}$$

**$F_1$ : precision & recall的调和平均值**

$$F_1 = \frac{2TP}{2TP + FP + FN}$$

$$\frac{2}{F_1} = \frac{1}{Precision} + \frac{1}{recall}$$



## 1.9 标注问题

**标注问题的输入是一个观测序列，输出是一个标记序列或状态序列**

评价标注模型的指标与评价分类模型的指标一样，常用的由**标注准确率、精确率、召回率**

统计学习方法： **隐马尔可夫模型、条件随机场**



## 1.10 回归问题

**回归问题用于预测输入变量和输出变量之间的关系**

*可以认为等价于函数拟合*

*一元回归 & 多元回归 | 线性回归 &非线性回归*

最常用的损失函数： **平方损失函数**

求解方法： **最小二乘法**



# 二 感知机

## 2.1 感知机模型

**定义**

$$ f(x) = sign(wx + b)$$

**感知机是一种线性分类模型，属于判别模型**

**超平面S** ： w x + b = 0 ;



## 2.2 感知机学习策略

**损失函数**的选择是误分点数到超平面S的总距离

定义为：

$$L(w,b) = - \sum y_i (w x_i + b)$$



## 2.3 感知机学习算法

最优化方法是**随机梯度下降法**



**公式推导：**

损失函数极小化的解 ： $$min_{w,b} L(w,b) = - \sum y_i(w x_i + b)$$

损失函数的梯度：

$$\nabla _w L(w,b) = - \sum y_ix_i$$

$$\nabla -b L(w,b) = - \sum y_i$$

更新权值和偏置：

$$w = w + \eta y_ix_i$$

$$b = b + \eta y_i$$



#### 感知机学习算法的原始形式

输入： 训练数据集T； 学习率$\eta$（0，1] 

输出： w, b ; 感知机模型 f(x) = sign( w x + b)

1. 选取初值$w_0,b_0$
2. 在训练集中选取数据（$w_i,y_i$)
3. 如果$y_i(w x_i + b) \leq 0$: 更新w ,b
4. 转至（2）， 直至训练集中没有误分类点



#### 算法的收敛性

当训练数据集线性可分时，感知机学习算法是收敛的。感知机算法在训练数据集上的误分类次数k满足不等式：$$k \leq (\frac{R}{\gamma})^2$$

当训练数据集线性可分时，感知机学习算法存在无穷多个解，其解由于不同的初值或不同的迭代顺序而可能有所不同。



#### 感知机学习算法的对偶形式

输入： 线性可分的数据集T ； 学习率 $\eta$ (0,1]

输出：$\alpha$ , b ; 感知机模型 $f(x) = sign( \sum _{j=1}^N \alpha _j y_j x_j x + b)$ ; 其中 $\alpha = (\alpha _1 ,\alpha _2 , ....., \alpha _n)^T$

1.  $\alpha$ = 0 , b = 0
2. 在训练集中选取数据$(x_i,y_i)$
3. 如果$y_i ( \sum _{j=1}^N \alpha _j y_j x_j x + b) \leq 0 $ : $$\alpha _i = \alpha _i + \eta$$ $$b = b + \eta y_i$$
4. 转至（2） 直至没有误分类数据出现



# 三 k 近邻法

k近邻法（KNN）是一种基本的**分类与回归方法**

KNN不具有显式的学习过程；

K值的选择、距离度量和分类决策规则时kNN的三个基本要素

## 3.1 k近邻算法

**算法：**

输入： 训练数据集 

输出： 实例x所属的类y

1. 根据给定的距离度量，在训练集T中找出与x最近邻的k个点，涵盖这k个点的x的邻域记作$N_k(x)$
2. 在$N_k(x)$中根据分类决策规则（如多数表决）决定x的类别y



## 3.2 k近邻模型

模型由3个基本要素——距离度量、k值的选择和分类决策规则决定的

#### 模型

**距离度量、k值的选择和分类决策规则**确定后，对于任何一个新的输入实例，它所属的类唯一地确定

#### 度量距离

$L_p$距离 or Minkowski距离

（p = 1 : 曼哈顿距离 ； p = 2： 欧式距离）

#### k值的选择

if k 较小，估计误差会增大，预测结果对近邻的实例点十分敏感； 即 k值的减小意味着整体模型变得复杂，容易发生过拟合；

if k 较大，学习的近似误差会增大； k值的增大意味着整体的模型会变得简单；

在应用中，k值一般选取一个比较小的数值，通常采用<font color = "red">**交叉验证**</font>的方法来选取最优的k值；

#### 分类决策规则

<font color = "red">**多数表决**</font>

*等价于风险最小化*



























































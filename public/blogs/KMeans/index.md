# 一、朴素贝叶斯公式的推导

## 1.1 基础知识铺垫

### 1.1.1 联合概率

**联合概率**是指**两个（或多个）事件同时发生**的概率。

用数学符号表示为： $P(A \cap B)$  或简写为  $P(A, B)$

联合概率着眼于**全集**，计算的是“既是 A 又是 B”占**总体**的比例

### 1.1.2 条件概率

**条件概率**是指在**事件 B 已经发生**的前提下，事件 A 发生的概率。

用数学符号表示为： $P(A \mid B)$

**关键**：条件概率改变了我们的**参考空间**。我们不再看全体样本，而是只看符合条件的那个子集。
$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \quad (P(B) > 0)
$$
只要把条件概率的公式稍微变形，就能得到联合概率与条件概率之间的桥梁：
$$
P(A \cap B) = P(A \mid B) \cdot P(B)
$$
同样地，也可以写成：
$$
P(A \cap B) = P(B \mid A) \cdot P(A)
$$

## 1.2 贝叶斯公式

$y$ 为类别标签， $x = (x_1, x_2, ..., x_d)$ 为样本的  $d$  个特征

**贝叶斯定理**给出了在已知特征  $x$  的条件下，样本属于类别  $y$  的概率：
$$
P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}
$$


|     符号      |     名称     |                           含义                            |
| :-----------: | :----------: | :-------------------------------------------------------: |
| $P(y \mid x)$ | **后验概率** | 看到样本特征 x  后，它属于类别 y 的概率（这是我们想要的） |
| $P(x \mid y)$ | **似然概率** |            在类别 y 中，样本呈现特征 x 的概率             |
|    $P(y)$     | **先验概率** |   在没有看到特征之前，类别  y 出现的概率，一般是目标值    |
|    $P(x)$     | **证据因子** |  特征 ( x ) 出现的边缘概率（归一化常数，对所有类别相同）  |

朴素贝叶斯的核心思想：给定一个样本 $x$，我们计算它属于每个类别的**后验概率** $P(y \mid x)$，然后选择**后验概率最大**的那个类别作为预测结果。

## 1.3 分类决策规则

对于分类问题，我们想要找到使后验概率最大的类别$\hat{y}$ ：
$$
\hat{y} = \arg\max_{y} P(y|x)
$$
由于分母 $P(x)$ 对于所有类别 $y$ 是**相同**的（都是同一个样本的特征分布概率)，在比较时可以忽略，因此分类决策简化为：
$$
\hat{y} = \arg\max_{y} \; P(x|y) \cdot P(y)
$$

## 1.4 朴素贝叶斯的假设

直接计算 $P(x \mid y)$是非常困难的。假设特征有 $d$ 个，每个特征有$v$ 种取值，那么联合概率空间大小为 $v^d$ ，在样本有限的情况下很难准确估计。

朴素贝叶斯做了一个**极强的假设**：**在给定类别 $y$ 的条件下，各个特征之间是相互独立的。**

用数学语言表达为：
$$
P(x|y) = P(x_1, x_2, ..., x_d \;|\; y) = \prod_{j=1}^{d} P(x_j \;|\; y)
$$
这个假设被称为**条件独立性假设**，也是“朴素”一词的来源

## 1.5 带入分类决策

将条件独立性假设代入决策规则：
$$
\hat{y} = \arg\max_{y} \; P(y) \prod_{j=1}^{d} P(x_j \;|\; y)
$$
这就是朴素贝叶斯分类器的核心公式。

## 1.6 先验概率$P(y)$的估计

在训练阶段，我们需要从训练数据中估计出公式中的两个核心部分：**先验概率**$P(y)$和**条件概率**$P(x_j | y)$

先验概率$P(y)$表示在没有任何特征信息的情况下，样本属于类别 $y$ 的概率。我们直接用训练集中各类别出现的频率来估计：
$$
P(y = c) = \frac{N_c}{N}
$$


| 符号  |              含义               |
| :---: | :-----------------------------: |
| $N_c$ | 训练集中属于类别 $c$ 的样本数量 |
|  $N$  |        训练集的总样本数         |

**推导**：这是二项分布/多项分布的**极大似然估计（MLE）**。假设类别服从类别分布，最大化对数似然 $\sum_{i} \log P(y_i)$，求导后解出的结果就是频率。

## 1.7 条件概率$P(x_j | y)$的估计

这是最关键的部分，根据**特征类型**（离散型或连续型）有不同的估计方法。

### 1.7.1 离散型特征

假设特征 $x_j$  有 $V$ 个可能的取值，我们直接用训练集中满足条件（类别为  $c$ ，且特征取值为  $v$ )的样本比例来估计：
$$
P(x_j = v \;|\; y = c) = \frac{N_{(c, v)}}{N_c}
$$


|     符号     |                           含义                           |
| :----------: | :------------------------------------------------------: |
| $N_{(c, v)}$ | 训练集中，类别为 c ，且第  j 个特征取值为  v  的样本数量 |
|   $N_{c}$    |              训练集中，类别为 c 的样本总数               |

### 1.7.2 连续型特征

如果特征是连续值（如身高、温度），我们通常假设特征在给定类别下服从**正态分布（高斯分布）**。

对于类别  c ，特征 $x_j$ 的概率密度函数为：
$$
P(x_j \;|\; y = c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\left( -\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right)
$$
其中：

$\mu_{jc}$ ：类别  c  中，第  j  个特征的样本均值 $\mu_{jc} = \frac{1}{N_c}\sum x_j$

$\sigma_{jc}^2$ ：类别  c  中，第  j  个特征的样本方差 $\sigma_{jc}^2 = \frac{1}{N_c}\sum (x_j - \mu_{jc})^2$

## 1.8 拉普拉斯平滑系数

如果某个特征值$x_j = v$在训练集中**没有**出现在类别  c  中，那么 $P(x_j=v | y=c) = 0$

根据我们的决策公式，只要有一个特征条件概率为 0，整个 $\prod_{j=1}^{d} P(x_j \;|\; y)$  就变成 0

这会导致模型将样本判为该类别的概率为 0，即使其他特征都非常匹配，也会被一票否决。

为了避免概率值为0，在分子分母同时加上一个小的常数$\alpha$（通常取 $\alpha = 1$ )：
$$
P(x_j = v \;|\; y = c) = \frac{N_{(c, v)} + \alpha}{N_c + \alpha \cdot V}
$$


|     符号     |                  含义                   |
| :----------: | :-------------------------------------: |
| $N_{(c, v)}$ | 类别  c 中，特征  j  取值为  v 的样本数 |
|    $N_c$     |           类别 c  的样本总数            |
|     $V$      |      第  j 个特征可能取值的总个数       |
|   $\alpha$   |    平滑系数通常取 1，即 Laplace 平滑    |

为什么分母是 $N_c + \alpha \cdot V$ ？

为了保证条件概率对所有 $V$ 个取值求和为 1：
$$
\sum_{v=1}^{V} \frac{N_{c, v} + \alpha}{N_c + \alpha \cdot V} = \frac{N_c + \alpha \cdot V}{N_c + \alpha \cdot V} = 1
$$


# 二、案例——商品评论情感分析

## 2.1 朴素贝叶斯API

![image-20260925141408520](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925141408520.png)

## 2.2 数据分析

![image-20260925141424296](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925141424296.png)

将内容进行分词，删除无效文字，

![image-20260925141436565](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925141436565.png)

## 2.3 代码实现

代码如下：

```python
import numpy as np  #数学计算包
import pandas as pd  #数据处理包
import matplotlib.pyplot as plt #可视化包
import jieba #中文分词包
from sklearn.feature_extraction.text import CountVectorizer #词频统计包 把评论内容 转成 词频矩阵 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB # 朴素贝叶斯分类器

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 读取数据
# encoding='gbk' 是为了解决中文乱码问题
data = pd.read_csv('./KMeans/data/BookEvaluate.csv', encoding='gbk') 

# 2. 数据的预处理
# 2.1 添加lables列 充当 标签列  好评：1 差评：0 
data['lables'] = np.where(data['评价'] == '好评', 1, 0)

# 2.2 拆取lables列，做为 标签
y = data['lables']

# 2.3 对用户的评论进行分词
# jieba.lcut(line) 对每一条评论进行分词 ','.join(jieba.lcut(line)) 把分词结果用逗号连接起来
comment_list = [','.join(jieba.lcut(line)) for line in data['内容']]

# 2.4 加载 停用词列表 即：里面记录的词，不需要参与模型训练
# with open() 代表 打开文件 as src_f 代表把打开的文件赋值给 src_f
with open('./KMeans/data/stopwords.txt', 'r', encoding='utf-8') as src_f:
    # 一次读取所有的行 
    stopwords_list = src_f.readlines()
    # 把读取的行中的换行符去掉
    stopwords_list = [line.strip() for line in stopwords_list]
    # 去重
    stopwords_list = list(set(stopwords_list))

# 2.5 把分词结果comment_list中的 停用词 去掉 并且统计词频
transfer = CountVectorizer(stop_words=stopwords_list) #参数 stop_words： 停用词列表

# 2.6 把词频矩阵 转成 特征矩阵 先训练，后转换，再转数组
x = transfer.fit_transform(comment_list).toarray() # 参数 toarray()： 转成数组

# 2.7 拆分数据集
x_train = x[:10]
y_train = y[:10]

x_test = x[10:]
y_test = y[10:]

# 3. 特征工程

# 4. 模型训练
estimator = MultinomialNB() # 创建朴素贝叶斯模型对象
estimator.fit(x, y) # 训练模型

# 5. 模型预测
y_pred = estimator.predict(x_test)

# 6. 模型评估
print(f'准确率：{accuracy_score(y_test, y_pred)}')
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

# 三、聚类算法

**KMeans** 是一种最经典、最常用的**无监督学习**算法，主要用于解决**聚类**问题。

其核心目标是：根据样本之间的**相似性**，将  n  个样本点划分到  K  个不同的簇中，使得每个样本点都属于离它最近的簇中心（质心）所代表的簇，同时让同一个簇内的样本点尽可能“紧密”地聚集在一起。

## 3.1 算法流程

KMeans 采用**坐标下降法**进行迭代求解，主要分为 4 步：

| 步骤 |      名称       |                             操作                             |
| :--: | :-------------: | :----------------------------------------------------------: |
|  1   |   **初始化**    | 随机选择 K 个样本点作为初始的簇中心( ![\mu_1, \mu_2, ..., \mu_K](https://latex.csdn.net/eq?%5Cmu_1%2C%20%5Cmu_2%2C%20...%2C%20%5Cmu_K) ) |
|  2   | **分配（E步）** | 计算每个样本点到 K 个质心的距离，将其划分到**距离最近**的质心所代表的簇中 |
|  3   | **更新（M步）** | 对每个簇，重新计算该簇内所有样本点的**均值**，将该均值作为该簇新的质心 |
|  4   |  **判断收敛**   | 如果质心不再发生变化（或变化小于预设阈值，或达到最大迭代次数），则算法结束；否则返回步骤 2 |

## 3.2 算法推导

在数学上，KMeans 试图最小化所有样本点到其所属簇中心之间的**平方距离之和**（也称为 **SSE：误差平方和**）。

设：

数据集 $D = {x_1, x_2, ..., x_n}$，每个样本 $x_i \in \mathbb{R}^d$

簇的数量为 K

第  j  个簇的质心为 $\mu_j$（这也是我们要求解的变量）

$r_{ij}$ 是指示变量：如果样本$x_i$属于簇 j ，则$r_{ij} = 1$ ，否则$r_{ij} = 1$

**目标函数（最小化）**：

![J = \sum_{i=1}^{n} \sum_{j=1}^{K} r_{ij} \|x_i - \mu_j\|^2](https://latex.csdn.net/eq?J%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Csum_%7Bj%3D1%7D%5E%7BK%7D%20r_%7Bij%7D%20%5C%7Cx_i%20-%20%5Cmu_j%5C%7C%5E2)

KMeans 要找到一组最优的质心 ![{\mu_j}](https://latex.csdn.net/eq?%7B%5Cmu_j%7D) 和最优的划分 ![r_{ij}](https://latex.csdn.net/eq?r_%7Bij%7D) ，让上面的 ![J](https://latex.csdn.net/eq?J) 尽可能小。

### 3.2.1 固定质心![{\mu_j}](https://latex.csdn.net/eq?%7B%5Cmu_j%7D)，优化划分![r_{ij}](https://latex.csdn.net/eq?r_%7Bij%7D)

假设当前的质心 ![{\mu_j}](https://latex.csdn.net/eq?%7B%5Cmu_j%7D) 是已知且固定的，我们想要决定每个样本  ![x_i](https://latex.csdn.net/eq?x_i)  应该属于哪个簇，即确定 ![r_{ij}](https://latex.csdn.net/eq?r_%7Bij%7D)

对于**单个样本** ![x_i](https://latex.csdn.net/eq?x_i) 来说，它对整体损失函数 ![J](https://latex.csdn.net/eq?J) 的贡献是：
$$
J = \sum_{i=1}^{n} \sum_{j=1}^{K} r_{ij} \|x_i - \mu_j\|^2
$$
由于约束是“每个样本只能属于一个簇”，即
$$
\sum_{j = 1}^{K} r_{ij} = 1
$$
为了让$J_i$最小，我们只需要在 K 个距离中选择**最小的那个**：

计算$x_i$到所有质心的距离：
$$
|x_i - \mu_1|^2, |x_i - \mu_2|^2, ..., |x_i - \mu_K|^2
$$
找到距离最小的质心，记其下标为 $j^*$

令$r_{ij^*} = 1$，其他 $r_{ij} = 0$

**结论**：在质心固定的情况下，最优的分配策略就是**将每个样本分配给离它最近的质心**（通常用欧氏距离）。

### 3.2.2 固定划分$r_{ij}$，优化质心${\mu_j}$

假设当前的样本划分已经固定（即每个样本属于哪个簇是确定的），我们想要找到每个簇的最优质心 ![\mu_j](https://latex.csdn.net/eq?%5Cmu_j)

由于划分固定，对于第  j  个簇来说，它包含的样本集合$S_{j}$ 是已知的。我们只需要单独最小化第  j  个簇的损失：
$$
J_j = \sum_{x_i \in S_j} \|x_i - \mu_j\|^2
$$
我们的目标是找到使$J_j$最小的$\mu_j$。由于$J_j$是关于 的$\mu_j$**凸函数**，我们只需对其求导，令导数为零即可。

设 $x_i$  和 $\mu_j$均为 $d$ 维列向量，即：
$$
x_i = \begin{bmatrix} x_{i1} \\ x_{i2} \\ \vdots \\ x_{id} \end{bmatrix}, \quad \mu_j = \begin{bmatrix} \mu_{j1} \\ \mu_{j2} \\ \vdots \\ \mu_{jd} \end{bmatrix}
$$
那么距离的平方为：
$$
\|x_i - \mu_j\|^2 = (x_i - \mu_j)^T (x_i - \mu_j) = \sum_{k=1}^{d} (x_{ik} - \mu_{jk})^2
$$
带入目标函数，有：
$$
J_j = \sum_{x_i \in S_j} \sum_{k=1}^{d} (x_{ik} - \mu_{jk})^2
$$
为了方便求导，我们需要对$\mu_j$的**每一个维度** $\mu_{jk}$ 分别求偏导。

只看$J_j$中含 $\mu_{jk}$的部分，只有$\sum_{k=1}^{d} (x_{ik} - \mu_{jk})^2$这一项包含它，其他项对$\mu_{jk}$求导为 0

对 $\mu_{jk}$ 求偏导（注意：对$\mu_{jk}$求导，$x_{ik}$是常数)：
$$
\frac{\partial J_j}{\partial \mu_{jk}} = \sum_{x_i \in S_j} \left[ -2 (x_{ik} - \mu_{jk}) \right]
$$
为了书写简洁，我们将其改写：
$$
\frac{\partial J_j}{\partial \mu_{jk}} = 2 \sum_{x_i \in S_j} (\mu_{jk} - x_{ik})
$$
为了找到极小值点，我们令偏导数为 0：
$$
2 \sum_{x_i \in S_j} (\mu_{jk} - x_{ik}) = 0
$$
两边同时除以 2，去掉常数系数：
$$
\sum_{x_i \in S_j} (\mu_{jk} - x_{ik}) = 0
$$
拆开求和符号：
$$
\sum_{x_i \in S_j} \mu_{jk} - \sum_{x_i \in S_j} x_{ik} = 0
$$
因为 $\mu_{jk}$ 是一个常数（对求和来说)，第一个求和项为：
$$
\sum_{x_i \in S_j} \mu_{jk} = n_j \cdot \mu_{jk}
$$
其中 $n_j = |S_j|$ 表示第  j  个簇中样本的数量。

代入等式得到：
$$
n_j \cdot \mu_{jk} - \sum_{x_i \in S_j} x_{ik} = 0
$$
解出最优质心第k个维度：
$$
n_j \cdot \mu_{jk} = \sum_{x_i \in S_j} x_{ik}
$$

$$
\mu_{jk} = \frac{1}{n_j} \sum_{x_i \in S_j} x_{ik}
$$

拼回完整的向量形式：

因为上面的推导对于$k = 1, 2, ..., d$都成立，我们将所有维度合并，得到：
$$
\mu_j = \begin{bmatrix} \mu_{j1} \\ \mu_{j2} \\ \vdots \\ \mu_{jd} \end{bmatrix} = \frac{1}{n_j} \begin{bmatrix} \sum x_{i1} \\ \sum x_{i2} \\ \vdots \\ \sum x_{id} \end{bmatrix} = \frac{1}{n_j} \sum_{x_i \in S_j} x_i
$$
**结论**：在划分固定的情况下，簇  j  的最优质心$\mu_j$就是该簇内所有样本点的**算术平均值**（Mean）

## **3.3** KMeans 算法的收敛性

由于每一轮迭代都执行了以下两个单调不增的步骤：

1. **分配步骤**：固定质心，重新分配样本到最近的质心，这必然使得目标函数 $J$ **不会增大**（因为每个样本都选择了使自己距离最小的簇）
2. **更新步骤**：固定划分，将质心更新为簇内均值，根据上面的数学推导，这**使得目标函数** $J$ **最小化**，因此也**不会增大**

由于 $J$ 有下界（最小为 0），且每一轮都在单调减小，所以 KMeans 算法**一定会收敛**。

**注意**：KMeans 收敛到的是**局部最优解**，而不是全局最优解。最终结果严重依赖于初始质心的选择。

## 3.4 SSE与肘部法

### 3.4.1 SSE

**SSE（Sum of Squared Errors，误差平方和）** 是衡量聚类效果最直接的指标之一，也是 KMeans 算法的**目标函数（损失函数）**。

它计算的是：**所有样本点到其所属簇中心（质心）的欧氏距离的平方和**。
$$
\text{SSE} = \sum_{i=1}^{n} \|x_i - \mu_{c_i}\|^2
$$


|    符号     |           含义           |
| :---------: | :----------------------: |
|      n      |         样本总数         |
|    $x_i$    |      第  i 个样本点      |
|    $c_i$    | 第 i 个样本所属的簇编号  |
| $\mu_{c_i}$ | 第 i  个样本所属簇的质心 |

SSE 衡量的是“所有样本点离自己所在的簇中心有多远”。SSE **越小**，说明样本点离簇中心越近，簇内紧密程度越高，聚类效果越好。

### 3.4.2 肘部法

当我们增加簇的数量  K  时，会发生两件事：

**SSE 一定会下降**：因为样本点有了更多簇可以选择，离自己最近的质心只会更近，不会更远。极端情况下，当  K = n （每个样本自成一簇）时，SSE = 0

**下降的速度在变慢**：从  K=1  到  K=2 ，SSE 会急剧下降（因为把一个松散的大簇分成了两个紧凑的小簇）。但随着  K  继续增加，SSE 下降的幅度会越来越小

**肘部法的原理**：绘制  K  值与 SSE 的关系曲线。曲线会呈现一个“胳膊肘”形状——先快速下降，然后变得平缓。**“肘部”拐点对应的  K  值，就是最佳的聚类数量。**

## 3.5 轮廓系数法

**轮廓系数法** 是评估聚类算法效果（内部聚类质量）最常用的指标之一，属于**无监督**评估方法。

它综合衡量了“簇内紧密度”与“簇间分离度”的比值，用于判断当前的聚类划分是否合理。

它同时兼顾了聚类的两个核心要求：

**簇内紧密（Cohesion）**：同一个簇内的样本点应该尽可能彼此靠近。

**簇间分离（Separation）**：不同簇之间的样本点应该尽可能远离。



对于数据中的 **单个样本点**  i ，其轮廓系数的计算需要以下两个值：

**簇内不相似度$a(i)$** 和**簇间不相似度$b(i)$**

### 3.5.1 **簇内不相似度**

**簇内不相似度$a(i)$** = 样本 i 到其所属簇内所有其他样本的**平均距离**

$a(i)$衡量的是样本  i 与**同簇**样本的紧密程度。我们希望它**越小越好**，表示簇内非常紧凑。

### 3.5.2 **簇间不相似度**

首先，对于样本 i  所在的簇$C_i$，我们观察其他的每一个簇$C$。计算样本  i  到簇  C  中**所有样本的平均距离**，然后取这些平均距离中的**最小值**：
$$
b(i) = \min_{C \neq C_i} (\text{样本 } i \text{ 到簇 } C \text{ 中所有样本的平均距离})
$$
**$b(i)$**衡量的是样本  i  与**离它最近的那个其它簇**的分离程度。我们希望它**越大越好**，表示样本离其它簇很远，不容易被分错。

### 3.5.3 单个样本的轮廓系数

在得到$a(i)$和$b(i)$后，单个样本的轮廓系数$s(i)$定义为：
$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$
$s(i)$衡量的是 “比该样本最近的其它簇，该样本更贴近自己的簇” 的程度。

### 3.5.4 整体轮廓系数

整个数据集的轮廓系数是**所有样本点轮廓系数的算术平均值**：
$$
\text{Silhouette\_Score} = \frac{1}{N} \sum_{i=1}^{N} s(i)
$$
该值即为最终评估聚类效果的指标。

### 3.5.5 轮廓系数取值

由于$a(i)$和$b(i)$都是距离（非负），轮廓系数$s(i)$的取值范围在 **[-1, 1]** 之间：

|  取值范围   |                             含义                             | 聚类质量 |
| :---------: | :----------------------------------------------------------: | :------: |
| **接近 +1** | $b(i) \gg a(i)$ ，即簇内距离**远小于**簇间距离。说明样本被分得很好，既贴合自己的簇，又远离其它簇。 | **优秀** |
| **接近 0**  | $a(i) \approx b(i)$，即簇内距离和最近的簇间距离差不多。说明样本处于两个簇的**边界**上，分类模糊。 | **一般** |
| **接近 -1** | $b(i) \ll a(i)$ ，即簇内距离**大于**簇间距离。说明样本点离其它簇比离自己的簇更近，属于**严重误分**。 |  **差**  |

### 3.5.6 用途

#### 3.5.6.1 选择最佳聚类数

这是最常用的场景。当我们不知道数据应该分成几类时，可以尝试多个 K 值，分别计算整体轮廓系数。

**选择准则**：使**轮廓系数最大**的  K  值对应的聚类效果最好。

**优势**：相比“肘部法则”（看SSE拐点），轮廓系数法通常更客观，不易受量纲影响。

#### 3.5.6.2 评价不同聚类算法

当使用 KMeans、层次聚类、DBSCAN 等不同算法得到不同的聚类结果时，可以比较它们的轮廓系数，选择系数更高的那个算法。

### 3.5.7 优缺点

|                             优点                             |                             缺点                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| **不依赖真实标签**：属于内部评估指标，不需要 Ground Truth，通用性强。 | **计算量大**：需要计算每个样本到所有其它样本的距离，复杂度约为  $O(N^2 \cdot d)$，在大数据集上较慢。 |
|    **指标直观**：值域固定在 [-1, 1]，便于理解和设定阈值。    | **对凸形簇敏感**：如果簇的形状不规则（如嵌套、细长链），即使聚类合理，轮廓系数也可能偏低。 |
| **可解释性强**：不仅能给出整体分数，还能分析每个样本的得分，识别出异常点或边界点。 | **依赖距离度量**：通常使用欧氏距离，若数据分布特殊，需谨慎选择其它距离度量。 |

### 3.5.8 肘部法与轮廓系数比较

|   对比维度   |                        肘部法（SSE）                         |                          轮廓系数法                          |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **核心指标** |                     簇内紧密程度（SSE）                      |                     簇内紧密 vs 簇间分离                     |
| **计算速度** | ✅ **快**（直接使用 KMeans 自带的 inertia 值，几乎无额外开销） | ❌ 较慢（需要计算所有样本点之间的距离，![O(N^2)](https://latex.csdn.net/eq?O%28N%5E2%29) |
| **结果形态** |            曲线下降图，寻找**拐点**（主观性较强）            |            分数图[-1,1]，寻找**峰值**（相对客观）            |
| **适用场景** |         快速初步确定 K 的范围；数据分布为凸形球状簇          |               需要精确比较模型质量；数据量适中               |
| **主要缺点** | 如果曲线平滑，没有明显的“肘部”，则无法判断（例如自然聚类不明显的数据集） |       对非凸形簇（如长条形、环状）效果较差；计算成本高       |

选择：

|                    场景                     |                      推荐方法                       |
| :-----------------------------------------: | :-------------------------------------------------: |
|          数据量巨大，想快速估计 K           |        优先使用**肘部法**（计算几乎零成本）         |
| 数据量中等，需要客观精确地对比 2~3 个候选 K |               优先使用**轮廓系数法**                |
|       肘部法曲线过于平滑，找不到拐点        | 改用**轮廓系数法**或**间隔统计量（Gap Statistic）** |

## 3.6 KMeans算法API

![image-20260925213756084](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925213756084.png)

## 3.7 案例——使用KMeans模型数据探索聚类

代码如下：

```python
import os
os.environ['OMP_NUM_THREADS'] = '3'

from sklearn.cluster import KMeans  # 聚类API 采用指定 质心 来分簇
import matplotlib.pyplot as plt     # 画图API
from sklearn.datasets import make_blobs  # 默认按照正态分布生成数据集，只需要指定 均值，标准差
from sklearn.metrics import calinski_harabasz_score  # 轮廓系数API 评价指标 值越大 聚类效果越好

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 准备数据集
# 参数1：样本数量 ，参数2：样本特征数量，参数3：样本标签数量，参数4：标准差，参数5：随机种子
x, y = make_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=1.0, random_state=23)

#plt.scatter(x[:, 0], x[:, 1], c=y)
#plt.show()

# 2. 数据的预处理
# 3. 特征工程，略

# 4. 模型训练
# 参数1：聚类数量，参数2：随机种子
estimator = KMeans(n_clusters=4, random_state=23)

# 5. 模型 训练和预测
y_pred = estimator.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

# 6. 模型评估
score = calinski_harabasz_score(x, y_pred)
print(f'评价指标：{score}')
```

## 3.8 评估指标演示

```python
import os
os.environ['OMP_NUM_THREADS'] = '4' # 设置OMP程序运行时使用的线程数

from sklearn.cluster import KMeans # 导入KMeans聚类算法
import matplotlib.pyplot as plt # 导入绘图库
from sklearn.datasets import make_blobs # 导入make_blobs函数
from sklearn.metrics import calinski_harabasz_score, silhouette_score # 导入Calinski-Harabasz指数

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字


# 1. 定义函数 演示SSE+肘部法
def sse():
    # 1.1 定义SSE列表，记录每个k值对应的SSE值
    sse_list = []

    # 1.2 生成数据集
    # 参数1：样本数量 ，参数2：特征数量，参数3：簇数，参数4：中心点标准差，参数5：随机种子
    x, y = make_blobs(n_samples=1000,
                    n_features=2, 
                    centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                    cluster_std=[0.4, 0.2, 0.2, 0.2],
                    random_state=23)

    # 1.3 遍历k值，计算每个k值对应的SSE值
    for k in range(1, 100):
        # 1.3.1 构建KMeans模型
        # 参数1：簇数，参数2：最大迭代次数，参数3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 1.3.2 模型训练
        estimator.fit(x)
        # 1.3.3 获取到每个簇的SSE值
        sse_value = estimator.inertia_
        # 1.3.4 将每个k值对应的SSE值记录到列表中
        sse_list.append(sse_value)
    
    # 1.4 绘制SSE曲线图
    plt.figure(figsize=(10, 5)) # 设置画布大小
    plt.title('SSE Value') # 设置标题
    plt.xticks(range(0, 100, 3)) # 设置x轴刻度 从0开始，步长为3
    plt.xlabel('k') # 设置x轴标签
    plt.ylabel('SSE') # 设置y轴标签
    plt.grid() # 显示网格线

    # 参数1 k值，参数2 SSE值
    plt.plot(range(1, 100), sse_list)
    plt.show()


# 2. 定义函数 演示轮廓系数SC
def sc():
    # 2.1 定义sc列表，记录每个k值对应的sc值
    sc_list = []
    
    # 2.2 生成数据集
    # 参数1：样本数量 ，参数2：特征数量，参数3：簇数，参数4：中心点标准差，参数5：随机种子
    x, y = make_blobs(n_samples=1000,
                    n_features=2, 
                    centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                    cluster_std=[0.4, 0.2, 0.2, 0.2],
                    random_state=23)
    
    # 2.3 遍历k值，计算每个k值对应的SSE值
    for k in range(2, 100): # 考虑簇外 至少两个簇
        # 2.3.1 构建KMeans模型
        # 参数1：簇数，参数2：最大迭代次数，参数3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 2.3.2 模型训练
        estimator.fit(x)
        # 2.3.3 模型预测
        y_pred = estimator.predict(x)

        # 2.3.4 获取到每个簇的sc值
        sc_value = silhouette_score(x, y_pred)

        # 2.3.5 将每个k值对应的SSE值记录到列表中
        sc_list.append(sc_value)
        
    # 2.4 绘制sc曲线图
    plt.figure(figsize=(10, 5)) # 设置画布大小
    plt.title('sc Value') # 设置标题
    plt.xticks(range(0, 100, 3)) # 设置x轴刻度 从0开始，步长为3
    plt.xlabel('k') # 设置x轴标签
    plt.ylabel('sc') # 设置y轴标签
    plt.grid() # 显示网格线
    
    # 参数1 k值，参数2 sc值
    plt.plot(range(2, 2 + len(sc_list)), sc_list)
    plt.show()

# 3. 定义函数 演示轮廓系数CH
def ch():
    # 3.1 定义sc列表，记录每个k值对应的sc值
    ch_list = []
         
    # 3.2 生成数据集
    # 参数1：样本数量 ，参数2：特征数量，参数3：簇数，参数4：中心点标准差，参数5：随机种子
    x, y = make_blobs(n_samples=1000,
                    n_features=2, 
                    centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                    cluster_std=[0.4, 0.2, 0.2, 0.2],
                    random_state=23)
         
    # 3.3 遍历k值，计算每个k值对应的SSE值
    for k in range(2, 100): # 考虑簇外 至少两个簇
        # 3.3.1 构建KMeans模型
        # 参数1：簇数，参数2：最大迭代次数，参数3：随机种子
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.3.2 模型训练
        estimator.fit(x)
        # 3.3.3 模型预测
        y_pred = estimator.predict(x)
     
        # 2.3.4 获取到每个簇的ch值
        ch_value = calinski_harabasz_score(x, y_pred)
     
        # 2.3.5 将每个k值对应的SSE值记录到列表中
        ch_list.append(ch_value)
             
    # 3.4 绘制sc曲线图
    plt.figure(figsize=(10, 5)) # 设置画布大小
    plt.title('ch Value') # 设置标题
    plt.xticks(range(0, 100, 3)) # 设置x轴刻度 从0开始，步长为3
    plt.xlabel('k') # 设置x轴标签
    plt.ylabel('ch') # 设置y轴标签
    plt.grid() # 显示网格线
         
    # 参数1 k值，参数2 sc值
    plt.plot(range(2, 2 + len(ch_list)), ch_list)
    plt.show()

# 4. 测试
if __name__ == '__main__':
    # sse()
    # sc()
    ch()
```

## 3.9 案例——客户案例分析

### 3.9.1 数据分析

![image-20260925213834712](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925213834712.png)

特征用第四列的年收入Annual Income，标签为消费指数Spending Score

### 3.9.2 代码实现

```python
import os
os.environ['OMP_NUM_THREADS'] = '4' # 设置OMP程序运行时使用的线程数

from sklearn.cluster import KMeans # 导入KMeans聚类算法
import matplotlib.pyplot as plt # 导入绘图库
from sklearn.metrics import calinski_harabasz_score, silhouette_score # 导入评估指标
import pandas as pd # 数据处理库

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 定义函数 找最优K值
def find_k():
    # 1.1 读取数据
    data = pd.read_csv('./KMeans/data/customers.csv')

    # 1.2 定义sse_list sc_list 记录不同k值的评估效果
    sse_list = []
    sc_list = []

    # 1.3 抽取特征
    x = data.iloc[:, 3:5]

    # 1.4 遍历k值
    for k in range(2, 20):
        # 1.4.1 创建KMeans模型
        kmeans = KMeans(n_clusters=k, max_iter=100, random_state=23)

        # 1.4.2 训练模型
        kmeans.fit(x)

        # 1.4.3 模型预测
        y_pred = kmeans.predict(x)

        # 1.4.4 把评分记录到对应列表
        sse_list.append(kmeans.inertia_)
        sc_list.append(silhouette_score(x, y_pred))

    # 1.5 绘制折线图 查看不同k值对应的sse和sc
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 20), sse_list, label='SSE')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 20), sc_list, label='SC')
    plt.show()

    # 结论 k=5 时 效果最好

# 2. 定义函数 训练预测评估
def train_predict_evaluate():
    # 2.1 读取数据
    data = pd.read_csv('./KMeans/data/customers.csv')
    
    # 2.2 抽取特征
    x = data.iloc[:, 3:5]

    # 2.3 创建KMeans模型 k = 5
    estimator = KMeans(n_clusters=5, max_iter=100, random_state=23)

    # 2.4 训练模型
    estimator.fit(x)

    # 2.5 模型预测
    y_pred = estimator.predict(x)

    # 2.6 绘制 5个簇的 样本点的 散点图
    plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y_pred)

    # 2.7 绘制 5个簇的 中心点
    plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], c='red', s=50)

    # 2.8 设置标题 和标签
    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')

    plt.show()


if __name__ == '__main__':
    # find_k()
    train_predict_evaluate()
```


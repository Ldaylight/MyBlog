# 一、神经网络介绍

**神经网络（Neural Network，NN）也叫 人工神经网络（Artificial Neural Network，ANN ）**是一种模仿生物神经网络结构和功能的**计算模型** 。它由大量相互连接的**神经元（Neuron）** 组成，通过调整连接之间的**权重（Weight）**，实现对复杂非线性函数的拟合。

## 1.1 神经网络

神经网络是由多个神经元组成，构建神经网络就是在构建神经元。神经元的构建说明如下：

![](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925221339031.png)

将输入层的数据进行加权求和，通过激活函数Sigmoid将结果映射到[0,1]之间

有几个特征，输入层就有几个神经元，同一层的多个神经元可以看作是通过并行计算来处理相同的输入数据。

**相邻层之间的神经元相互连接**，第N层的每个神经元和第N-1层的所有神经元相连，这就是**全连接神经网络（FCNN）。**全连接神经网络接收的样本数据是**二维的**，数据在每一层之间需要以二维的形式传递，每个连接都有一个权重值（w系数和b系数）

**同一层神经元相互隔离没有连接**，前一层的输出 = 后一层的输入

使用多个神经元来构建神经网络，给每一个连接分配一个强度w，如下图所示：

![image-20260925221400229](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925221400229.png)

神经网络中信息只向一个方向移动，即从输入节点向前移动，通过隐藏节点，再向输出节点移动。其中的基本部分是:

**输入层（Input Layer）**: 即输入x的那一层。每个输入特征对应一个神经元。不进行计算，仅传递数据 。输入层将数据传递给下一层的神经元。

**输出层（Output Layer）**: 即输出y的那一层。输出最终预测结果（类别/数值），结构取决于任务：回归、分类

**隐藏层（Hidden Layers）**: 输入层和输出层之间都是隐藏层，神经网络的“深度”通常由隐藏层的数量决定。自动提取特征、进行非线性变换 ，层数越多，网络越“深”，表达能力越强。

# 二、激活函数

**激活函数（Activation Function）** 是神经网络中的一种数学函数，它接收上一层神经元的加权输入，并**决定该神经元是否应该被“激活”**（即是否将信号传递给下一层）, 进而为整个网络注入了**非线性因素**。此时, 神经网络就可以拟合各种曲线，这样就可以来做分类问题了。

如果没有激活函数，无论神经网络有多少层，最终都等价于一个线性变换。这将使网络完全失效。

举例如下：

![image-20260925221747338](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925221747338.png)

每一个神经元都有自己的权重（指向自己的箭头上的权重）

![image-20260925221836314](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925221836314.png)

将上面两个公式带入下面计算：

![image-20260925221905478](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925221905478.png)

结果还是线性的

## 2.1 常见的激活函数

### 2.1.1 sigmoid激活函数

sigmoid激活函数公式：
$$
f(x) = \frac{1}{1 + e^{-x}}
$$
求导：
$$
f'(x) = f(x)(1 - f(x))
$$
函数图像如下：

![image-20260925222103913](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925222103913.png)

优点：

|       优点       |                             说明                             |
| :--------------: | :----------------------------------------------------------: |
|  **平滑且可导**  |       处处可微，适合基于梯度的优化算法（SGD、Adam 等）       |
| **输出范围固定** |           输出在 (0, 1) 之间，非常适合**表示概率**           |
|   **导数简单**   | ![f'(x) = f(x)(1 - f(x))](https://latex.csdn.net/eq?f%27%28x%29%20%3D%20f%28x%29%281%20-%20f%28x%29%29)，反向传播计算高效 |
|   **解释性强**   |       二分类任务中，输出可以直接解释为“属于正类的概率”       |

缺点：

|      缺点      |                             说明                             |                             后果                             |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  **梯度消失**  |             当输入非常大或非常小时，梯度趋近于 0             | 深层网络中，梯度在反向传播时指数级衰减，导致靠前的层几乎无法更新 |
|  **非零中心**  |                   输出均值恒为 0.5，而非 0                   | 梯度更新时，所有权重的梯度符号相同，导致 **“Z 型更新”**（zig-zag），收敛变慢 |
| **计算开销大** | 涉及指数运算 ![e^{-x}](https://latex.csdn.net/eq?e%5E%7B-x%7D) |               相比 ReLU 的 `max(0, x)` 更耗时                |
|   **软饱和**   |               即使输入变化很大，输出变化也很小               |           神经元在饱和区对输入不敏感，信息流动受阻           |

使用场景：

|         场景         |                           原因                            |
| :------------------: | :-------------------------------------------------------: |
|   **二分类输出层**   |             输出在 (0,1)，可直接作为正类概率              |
| **多标签分类输出层** | 每个神经元独立输出 0~1 概率（一个样本可同时属于多个类别） |
|    **注意力机制**    |                 生成 0~1 之间的注意力权重                 |
|     **门控机制**     |   LSTM / GRU 中的“遗忘门”、“输入门”等，控制信息流动比例   |
|     **概率映射**     |     将任意数值映射为概率值（如强化学习中的策略网络）      |

### 2.1.2 tanh激活函数

**Tanh（Hyperbolic Tangent，双曲正切）** 函数是神经网络中经典的激活函数之一。它是 Sigmoid 函数的**缩放平移版本**，将任意实数输入映射到 **(-1, 1)** 区间，且以 0 为中心。

Tanh激活函数公式如下：
$$
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{2}{1 + e^{-2x}} - 1
$$
Tanh 的导数如下：
$$
f'(x) = 1 - f^2(x)
$$
![image-20260925222228705](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925222228705.png)

优点

|      优点      |                      说明                       |
| :------------: | :---------------------------------------------: |
|   **零中心**   |       输出均值为 0，收敛速度比 Sigmoid 快       |
|  **梯度更大**  | 最大导数为 1（Sigmoid 仅 0.25），梯度传播更顺畅 |
| **平滑且可导** |        处处可微，适合基于梯度的优化算法         |
|  **导数简单**  |     $f'(x) = 1 - f^2(x)$，反向传播计算高效      |
|   **对称性**   |    奇函数：$f(-x) = -f(x)$，利于建模正负信号    |

缺点：

|       缺点       |               说明               |                      后果                       |
| :--------------: | :------------------------------: | :---------------------------------------------: |
|   **梯度消失**   | 当输入过大或过小时，梯度趋近于 0 | 深层网络中仍存在梯度消失问题（但比 Sigmoid 轻） |
| **计算开销较大** |  涉及指数运算 $e^x$ 和 $e^{-x}$  |              慢于 ReLU 的简单比较               |
|    **软饱和**    |     在两端（±∞）对输入不敏感     |                  信息流动受阻                   |

使用场景

|           场景           |                      原因                       |
| :----------------------: | :---------------------------------------------: |
|  **RNN / LSTM 隐藏层**   | 零中心、梯度比 Sigmoid 更稳定，适合处理时间序列 |
|   **GAN 生成器输出层**   |     输出范围 (-1, 1)，与图像归一化范围匹配      |
| **需要零中心数据的场景** |          输出均值为 0，利于下一层学习           |
|    **早期的浅层网络**    |    在 ReLU 流行之前，Tanh 是隐藏层的常用选择    |
|     **特征标准化层**     |            将特征映射到 (-1, 1) 区间            |

### 2.1.3 ReLU激活函数

**ReLU（Rectified Linear Unit，修正线性单元）** 是当今深度学习中**最主流、最常用**的激活函数。它简单到极致——正数保留，负数归零。

公式如下：
$$
f(x) = max(0, x)
$$
ReLU 在 ( x=0 ) 处不可导，但工程上通常约定其导数为：
$$
f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}
$$
函数(左)和导数(右)图像如下：

![image-20260925222826786](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925222826786.png)

优点

|       优点       |                            说明                             |
| :--------------: | :---------------------------------------------------------: |
|   **计算极快**   | 只需比较 $\max(0, x)$ ，无需指数运算（远快于 Sigmoid/Tanh） |
| **缓解梯度消失** |    正区间梯度恒为 1，不会像 Sigmoid/Tanh 那样在两端衰减     |
|   **稀疏激活**   |      负数全部归零，约 50% 的神经元输出为 0，减少计算量      |
|  **促进稀疏性**  |   稀疏性意味着更少的参数参与计算，有助于提取更鲁棒的特征    |
|  **生物合理性**  |           更接近生物神经元的“全有或全无”发放模式            |

缺点

|             缺点             |                       说明                       |                 后果                 |
| :--------------------------: | :----------------------------------------------: | :----------------------------------: |
| **神经元死亡（Dying ReLU）** | 如果输入始终为负，梯度为 0，神经元将永远无法激活 |    大量死亡神经元导致模型容量下降    |
|         **非零中心**         |          输出均值为正（约 0.5），而非 0          |        可能导致梯度更新不稳定        |
|         **负值截断**         |                 负值信息完全丢失                 | 无法表达“负信号”的意义（如抑制效果） |

**神经元死亡：**当一个神经元的输入 x 在训练中始终小于 0，它的梯度恒为 0，权重永远无法更新。这个神经元就“死了”——它对任何输入都输出 0。

### 2.1.4 SoftMax激活函数

**Softmax** 是一种专门用于**多分类**任务的激活函数，通常放在神经网络的**输出层**。它将一个 K 维的实数向量$z_1, z_2, ..., z_K$转换为一个**概率分布**——每个输出值都在 (0, 1)之间，且所有输出之和为 1。

SoftMax激活函数公式如下：
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, 2, ..., K
$$

|           符号           |               含义               |
| :----------------------: | :------------------------------: |
|           $K$            |             类别总数             |
|          $z_i$           | 第 $i$ 个类别的原始输出（logit） |
|        $e^{z_i}$         |  对 logit 取指数（保证为正数）   |
| $\sum_{j=1}^{K} e^{z_j}$ | 所有类别的指数之和（归一化分母） |

举例：

![image-20260925223732261](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925223732261.png)

### 2.1.5 其他激活函数

![image-20260925223833909](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925223833909.png)

## 2.2 激活函数的选择

|    位置    |      场景      |          首选           |    备选     |
| :--------: | :------------: | :---------------------: | :---------: |
| **隐藏层** |    默认选择    |        **ReLU**         | Leaky ReLU  |
|            | 神经元死亡严重 |     **Leaky ReLU**      | PReLU / ELU |
|            |  Transformer   |        **GELU**         |    Swish    |
|            |   RNN / LSTM   |        **Tanh**         |     ELU     |
|            |   GAN 判别器   |     **Leaky ReLU**      |    ReLU     |
| **输出层** |     二分类     |       **Sigmoid**       |      —      |
|            | 多分类（互斥） |       **Softmax**       |      —      |
|            |   多标签分类   | **Sigmoid**（每个节点） |      —      |
|            |      回归      |     **无（线性）**      |      —      |

# 三、参数初始化

我们在构建网络之后，网络中的参数是需要初始化的。我们需要初始化的参数主要有**权重**和**偏置**，**偏置一般初始化为0即可**，而对权重的初始化则会更加重要。

参数初始化就是“给模型一个**起点**”，决定了模型从何处开始寻找最优解。

|   初始化质量   |                         对训练的影响                         |
| :------------: | :----------------------------------------------------------: |
| **好的初始化** |      模型从一开始就处于“合适的区域”，梯度适中，收敛快速      |
| **坏的初始化** | 模型起点偏差过大，梯度消失或爆炸，需要极长的训练时间甚至无法收敛 |

对梯度的影响：

|          问题          |                 与初始化的关系                 |
| :--------------------: | :--------------------------------------------: |
|      **梯度消失**      | 权重初始化太小 → 信号逐层衰减 → 梯度指数级变小 |
|      **梯度爆炸**      | 权重初始化太大 → 信号逐层放大 → 梯度指数级变大 |
| **神经元死亡（ReLU）** | 权重初始化不当 → 大量输入落入负区间 → 梯度为 0 |

打破对称性：

|            情况            |                             后果                             |
| :------------------------: | :----------------------------------------------------------: |
| **所有权重初始化为相同值** | 同一层的所有神经元学习完全相同的特征，相当于“只有一个神经元”在工作 |
| **随机初始化（打破对称）** |     每个神经元朝向不同的方向学习，分工合作，提高模型容量     |

## 3.1 参数初始化方法

|                      初始化方法                       |                 初始化方式                 |                       核心公式 / 参数                        |                        优点                        |                           缺点                           |                        适用场景                        |
| :---------------------------------------------------: | :----------------------------------------: | :----------------------------------------------------------: | :------------------------------------------------: | :------------------------------------------------------: | :----------------------------------------------------: |
|               **随机初始化(均匀分布)**                |            从均匀分布中随机采样            | $W \sim \mathcal{U}\left(-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}}\right)$， d 为输入神经元数量 |                  能有效打破对称性                  |          随机范围选择不当可能导致梯度消失/爆炸           |         浅层网络（1-3层隐藏层）或低复杂度模型          |
|              **随机初始化（正态分布）**               | 从高斯分布中随机采样（通常取很小的标准差） | $W \sim \mathcal{N}(0, \sigma^2)$，常用 $\sigma = 0.01$ 或 根据层维度自适应缩放 |                  能有效打破对称性                  |              标准差选择不当可能导致梯度问题              |                 浅层网络或低复杂度模型                 |
|                     **全0初始化**                     |               所有权重设为 0               |                           $W = 0$                            |                      实现简单                      | **无法打破对称性**，所有神经元同步更新，等价于单一神经元 |           几乎不用于权重；仅用于偏置项初始化           |
|                     **全1初始化**                     |               所有权重设为 1               |                           $W = 1$                            |                      实现简单                      |      ① 无法打破对称性 ② 激活值指数级增长 → 梯度爆炸      |            几乎不用于权重；仅用于测试/调试             |
|                   **固定值初始化**                    |          所有权重设为同一个常数 c          |                   $W = c$（ c 为任意常数）                   |                      实现简单                      |      ① 无法打破对称性 ② 值过大/过小 → 梯度爆炸/消失      |            几乎不用于权重；仅用于测试/调试             |
| **Xavier / Glorot 初始化** 			**(正态分布)** |   从高斯分布中采样，方差基于输入输出维度   | $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$ | **保持前向和反向传播的方差一致**，有效缓解梯度消失 |  对 ReLU 激活函数**效果欠佳**（未考虑 ReLU 的方差减半）  |  深度网络（≥10层），使用 **Sigmoid / Tanh** 激活函数   |
| **Xavier / Glorot 初始化** 			**(均匀分布)** |   从均匀分布中采样，边界基于输入输出维度   | $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$ |                        同上                        |                           同上                           |                          同上                          |
|  **Kaiming / He 初始化** 			**(正态分布)**   |     从高斯分布中采样，方差基于输入维度     | $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$  |     **专为 ReLU 设计**，补偿方差减半，梯度稳定     |                对非 ReLU 激活函数效果一般                | 深度网络（≥10层），使用 **ReLU / Leaky ReLU** 激活函数 |
|  **Kaiming / He 初始化** 			**(均匀分布)**   |     从均匀分布中采样，边界基于输入维度     | $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$ |                        同上                        |                           同上                           |                          同上                          |

### 3.1.1 Kaiming / He 初始化

**提出者**：何恺明 et al.（2015年提出）

**适用**：**ReLU 及其变体**（如 Leaky ReLU）

**目标**：解决 ReLU 将负值置零导致的方差损失问题

**公式**（正态分布版本，最常用）：
$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
$$
**公式**（均匀分布版本）：
$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}} \right)
$$
$n_{in}$表示当前层**输入神经元**的数量

### 3.1.2 **Xavier / Glorot**

**Xavier 初始化**（也称 **Glorot 初始化**）是一种权重初始化方法。它是专门针对 **Sigmoid** 和 **Tanh** 等“S 型”激活函数设计的，旨在解决深层网络中的**梯度消失**和**梯度爆炸**问题。

|   分布类型   |                             公式                             |                  标准差 / 范围                  |       适用场景       |
| :----------: | :----------------------------------------------------------: | :---------------------------------------------: | :------------------: |
| **正态分布** | $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$ |  $\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$   |    通用，推荐使用    |
| **均匀分布** | $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n{in} + n{out}}}, \sqrt{\frac{6}{n{in} + n{out}}}\right)$ | 边界$( \pm \sqrt{\frac{6}{n_{in} + n_{out}}} )$ | PyTorch 默认实现之一 |

$n_{in}$当前层**输入神经元**的数量 ，$n_{out}$当前层**输出神经元**的数量

示例代码：

```python
import torch.nn as nn


# 1. 均匀分布随机初始化
def test01():

    linear = nn.Linear(5, 3)
    # 从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)


# 2. 固定初始化
def test02():

    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 5)
    print(linear.weight.data)


# 3. 全0初始化
def test03():

    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    print(linear.weight.data)


# 4. 全1初始化
def test04():

    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    print(linear.weight.data)


# 5. 正态分布随机初始化
def test05():

    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight, mean=0, std=1)
    print(linear.weight.data)


# 6. kaiming 初始化
def test06():

    # kaiming 正态分布初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
    print(linear.weight.data)

    # kaiming 均匀分布初始化
    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    print(linear.weight.data)


# 7. xavier 初始化
def test07():

    # xavier 正态分布初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight.data)

    # xavier 均匀分布初始化
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)
```

# 四、搭建神经网络

先安装一个包torchsummary， 是一个用于查看 PyTorch 模型结构和参数数量的轻量级工具，方便调试和优化模型。

点开 命令行 或者 anacoda 的 prompt，输入如下代码：

```bash
pip install torchsummary -i https://mirrors.aliyun.com/pypi/simple/
```

## 4.1 搭建神经网络的过程

### 4.1.1 数据准备

|       子步骤        |                      操作                      |                      代码示例                      |
| :-----------------: | :--------------------------------------------: | :------------------------------------------------: |
|    1.1 数据加载     |        加载原始数据（CSV、图像、音频）         |          `data = pd.read_csv('data.csv')`          |
|   1.2 数据预处理    |      标准化/归一化、缺失值处理、数据增强       |        `transform = transforms.ToTensor()`         |
|   1.3 数据集划分    |           分为训练集、验证集、测试集           |      `train_test_split(X, y, test_size=0.2)`       |
| 1.4 封装 DataLoader | 打包为 `Dataset` 和 `DataLoader`，支持批量迭代 | `DataLoader(dataset, batch_size=64, shuffle=True)` |

### 4.1.2 模型构建

|      子步骤      |                            操作                            |             代码示例             |
| :--------------: | :--------------------------------------------------------: | :------------------------------: |
|  2.1 定义层结构  |  在 `__init__` 中声明所有层（Linear、Conv2d、Dropout 等）  | `self.fc1 = nn.Linear(784, 256)` |
| 2.2 定义前向传播 | 在 `forward` 中编写数据流动逻辑（层与层的连接 + 激活函数） |  `x = torch.relu(self.fc1(x))`   |

### 4.1.3 训练配置

|             组件              |           作用           |                           代码示例                           |
| :---------------------------: | :----------------------: | :----------------------------------------------------------: |
|   **损失函数（Criterion）**   | 衡量预测值与真实值的差距 |   `nn.CrossEntropyLoss()`（分类） / `nn.MSELoss()`（回归）   |
|    **优化器（Optimizer）**    |     根据梯度更新参数     |          `optim.Adam(model.parameters(), lr=0.001)`          |
| **学习率调度器（Scheduler）** |  动态调整学习率（可选）  | `optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)` |
|      **设备（Device）**       |     指定 CPU 或 GPU      | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |

### 4.1.4 训练循环

|         步骤          |              操作              |              关键代码               |
| :-------------------: | :----------------------------: | :---------------------------------: |
|     4.1 清空梯度      |      防止上一轮的梯度累积      |       `optimizer.zero_grad()`       |
|     4.2 前向传播      |   将数据传入模型得到预测结果   |      `outputs = model(inputs)`      |
|     4.3 计算损失      | 用损失函数计算预测与真值的误差 | `loss = criterion(outputs, labels)` |
|     4.4 反向传播      |        计算当前梯度的值        |          `loss.backward()`          |
|     4.5 更新参数      |      根据梯度下降更新权重      |         `optimizer.step()`          |
| 4.6（可选）调整学习率 |          按调度器更新          |         `scheduler.step()`          |

### 4.1.5 验证与评估

|      操作       |                     代码                      |                  说明                  |
| :-------------: | :-------------------------------------------: | :------------------------------------: |
| 切换到评估模式  |                `model.eval()`                 | 关闭 Dropout，BatchNorm 使用全局统计量 |
|  禁用梯度计算   |            `with torch.no_grad():`            | 节省显存和计算时间，推理时不需计算梯度 |
| 计算准确率/指标 | `accuracy = (preds == labels).float().mean()` |       根据任务选择合适的评估指标       |

### 4.1.6 保存与加载

|          场景          |               推荐方式               |                     代码示例                     |
| :--------------------: | :----------------------------------: | :----------------------------------------------: |
| **仅保存权重（推荐）** |    `state_dict`（体积小，恢复快）    |  `torch.save(model.state_dict(), 'model.pth')`   |
| 保存整个模型（不推荐） | 序列化整个对象（体积大，版本易出错） |         `torch.save(model, 'model.pth')`         |
|        加载权重        |       先实例化模型，再加载权重       | `model.load_state_dict(torch.load('model.pth'))` |

# 4.2 案例

构建如图所示的神经网络

![image-20260926085405216](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260926085405216.png)

编码设计如下：

第1个隐藏层：权重初始化采用标准化的xavier初始化 激活函数使用sigmoid

第2个隐藏层：权重初始化采用标准化的He初始化 激活函数采用relu

out输出层线性层 假若多分类，采用softmax做数据归一化

代码如下：

```python
import torch
import torch.nn as nn
from torchsummary import summary # 计算模型参数,查看模型结构

#VS Code 终端默认编码不是 UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 创建神经网络模型类 自定义继承 nn.Module
class Model(nn.Module):

    # 1. 在init()方法中完成初始化：父类成员 和 神经网络的搭建
    def __init__(self):

        # 1.1 初始化父类成员 # 调用父类的初始化属性值，确保nn.Module的初始化代码能够正确执行
        super(Model, self).__init__()

        # 1.2 搭建神经网络的隐藏层、输出层
        # 1.2.1 创建第一个隐藏层模型, 输入特征：3, 输出特征：3, 激活函数：Sigmoid
        self.linear1 = nn.Linear(3, 3)

        # 1.2.2 创建第二个隐藏层模型, 输入特征：3, 输出特征：2, 激活函数：ReLU
        self.linear2 = nn.Linear(3, 2)

        # 1.2.3 创建输出层模型, 输入特征：2, 输出特征：2, 激活函数：Softmax
        self.out = nn.Linear(2, 2)

        # 1.3 对隐藏层进行参数初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)


    # 2. 创建前向传播方法, 从输入层->隐藏层->输出层 
    # 函数名称必须叫forward, 调用神经网络模型对象时自动执行forward()方法
    def forward(self, x):
        # 2.1 第一层 隐藏层计算：加权求和 + 激活函数(Sigmoid)
        x = self.linear1(x)     # 通过隐藏层1 进行了加权求和
        x = torch.sigmoid(x)    # 使用sigmoid激活函数
        # x = torch.sigmoid(self.linear1(x)) 合并写法

        # 2.2 第二层 隐藏层计算：加权求和 + 激活函数(ReLU)
        x = torch.relu(self.linear2(x))

        # 2.3 输出层计算：加权求和 + 激活函数(Softmax)
        x = torch.softmax(self.out(x), dim=-1) # dim=-1 表示按照最后一个维度进行计算

        return x


# 模型训练
def train():
    
    # [修改] 添加：获取当前可用设备（如果有 GPU 则使用 GPU，否则使用 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 实例化model对象
    my_model = Model()
    my_model = my_model.to(device)# 添加：将模型移动到指定设备

    # 2. 创建数据集样本 随机生成
    data = torch.randn(5, 3) # 5个样本，每个样本3个特征
    data = data.to(device)# 添加：将数据也移动到同一设备
    print("data-->", data)
    print("data shape", data.shape)

    # 3. 调用神经网络模型 进行模型训练
    output = my_model(data) # 调用神经网络模型对象，自动执行forward()方法 进行前向传播
    print("output-->", output)
    print("output shape-->", output.shape)

    # 4. 计算 和 查看 模型参数
    # 计算每层每个神经元的w和b个数总和
    print("======计算模型参数======")
    #  因为 torchsummary 默认在 GPU 上创建输入（如果 cuda 可用），但模型已在 CPU 上，
    #  所以需要强制指定 device='cpu'，同时将模型临时移到 CPU。
    my_model_cpu = my_model.to("cpu")

    # 参数1: 模型对象 参数2: 输入特征维度 参数3: 批次大小
    summary(my_model_cpu, input_size=(3,), batch_size=5, device='cpu')
    
    my_model = my_model_cpu.to(device)# summary 之后把模型移回原来的设备，保持一致性
    
    # 5. 查看模型参数
    print("======查看模型参数w和b======")
    for name, param in my_model.named_parameters():
        print(name, param)

# 测试
if __name__ == '__main__':
    train()
```

# 五、损失函数

损失函数在不同的文献中名称是不一样的，主要有以下几种命名方式：

![image-20260926085458278](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260926085458278.png)

在深度学习中, 损失函数是用来衡量模型参数质量的函数, 衡量的方式是比较网络输出（预测值）和

真实输出（真实值）的差异。

模型通过**最小化损失函数**的值来调整参数，使其输出更接近真实值。

## 5.1 分类任务损失函数

### 5.1.1 二分类交叉熵

在处理二分类任务时，我们使用sigmoid激活函数，损失函数使用二分类交叉熵损失函数，计算公

式为：
$$
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$


|   **符号**    |            **含义**            |
| :-----------: | :----------------------------: |
|      $n$      |            样本总数            |
|    $y_{i}$    | 第i个样本是否属于类别c的真实值 |
| $\hat{y}_{i}$ |  模型预测这个样本为正类的概率  |

#### 5.1.1.1 推导过程

对于第 i 个样本，真实标签为$y_{i}$, 模型预测这个样本i的正类概率为$\hat{y}_{i}$

模型认为该样本i属于真实类别的概率可以统一写成：
$$
P(y | \hat{y}) = \hat{y}^y \cdot (1 - \hat{y})^{(1-y)}
$$
假设所有样本**独立同分布**，整个数据集的 **联合概率（似然函数）**为：
$$
L = \prod_{i=1}^{n} \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1-y_i}
$$
取负对数：
$$
-\log L = -\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$
求不求平均都可以，PyTorch默认取平均

#### 5.1.1.2 基本代码实现

在PyTorch中实现时使用nn.BCELoss()实现

```python
import torch
from torch import nn
def test02():
    # 1 设置真实值和预测值
    y_true = torch.tensor([0, 1, 0], dtype=torch.float32)
    # 预测值是sigmoid输出的结果
    y_pred = torch.tensor([0.6901, 0.5459, 0.2469], requires_grad=True)
    #2 二分类交叉熵损失
    loss = nn.BCELoss()
    #3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('loss：', my_loss)

if __name__ == '__main__':
    test02()
```

### **5.1.2 多分类交叉熵**

在多分类任务通常使用**softmax**将logits(原始得分)转换为概率的形式，所以多分类的交叉熵损失也

叫做softmax损失，它的计算公式是：
$$
\text{CE} = -\sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$


|    **符号**    |            **含义**            |
| :------------: | :----------------------------: |
|      $n$       |            样本总数            |
|      $c$       |              类别              |
|    $y_{ic}$    | 第i个样本是否属于类别c的真实值 |
| $\hat{y}_{ic}$ | 模型预测这个样本i为类别c的概率 |

场景：多分类（手写数字识别、物体分类）。

#### 5.1.2.1 推导过程

多分类是二分类的推广。真实标签服从类别分布（Categorical Distribution）

对于第 i 个样本，真实标签为 $y_{i1}, y_{i2}, ..., y_{iC}$，模型预测这个样本为类别c的概率为

$\hat{y}_{i1}, \hat{y}_{i2}, ..., \hat{y}_{iC}$，所有类别的概率之和为 1

模型认为该样本$i$属于真实类别$c$（一种类别的概率)的概率可以统一写成：
$$
P(y_{ic} | \hat{y}_{ic}) = \hat{y_{ic}}^{y_{ic}}
$$
一共有C类，就有C种可能。则该模型认为该样本属于真实类别（所有类别）的概率可以统一写

成：
$$
P(y_i | \hat{y_i}) = \prod_{c=1}^{C} (\hat{y}_{ic})^{y_{ic}}
$$
假设所有样本独立同分布，所有样本同时属于各自真实类别的概率（联合概率/似然函数）为：
$$
L(\theta) = \prod_{i=1}^{n} \prod_{c=1}^{C} (\hat{y}_{ic})^{y_{ic}}
$$
对似然函数取自然对数：
$$
\log L(\theta) = \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \cdot \log(\hat{y}_{ic})
$$
我们想让模型在训练数据上的**似然最大化（即 $logL$ 最大）**，但优化算法习惯做“最小化”。所以取

负号：
$$
\text{Loss} = -\log L(\theta) = -\sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \cdot \log(\hat{y}_{ic})
$$
推导完毕

#### 5.1.2.2 基本代码实现

```python
import torch
from torch import nn

#VS Code 终端默认编码不是 UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 分类损失函数：交叉熵损失使用nn.CrossEntropyLoss()实现。nn.CrossEntropyLoss()=softmax+损失计算
def test01():
    #设置真实值 注意：类型必须是64位整型数据
    y_true = torch.tensor([1, 2], dtype=torch.int64)

    #设置预测值
    y_pred = torch.tensor([[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]], requires_grad=True, dtype=torch.float32)

    #创建交叉熵损失函数，默认求平均损失
    loss = nn.CrossEntropyLoss()

    #计算损失结果
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('loss:', my_loss)

if __name__ == '__main__':
    test01()
```

## 5.2 回归任务损失函数

### 5.2.1 均方误差MSE

**Mean Squared Loss/ Quadratic Loss(MSE loss)**也被称为L2 loss，或欧氏距离，它以**误差的平方和**的均值作为距离

损失函数公式：
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

|   **符号**    |     **含义**      |
| :-----------: | :---------------: |
|      $n$      |     样本总数      |
|    $y_{i}$    | 第i个样本的真实值 |
| $\hat{y}_{i}$ | 第i个样本的预测值 |

优缺点

|               优点               |                         缺点                         |                适用场景                |
| :------------------------------: | :--------------------------------------------------: | :------------------------------------: |
| **处处可导**，梯度稳定，收敛平滑 | 对**异常值（Outliers）极其敏感**（误差平方放大偏差） | 数据干净、噪声较小、无极端值的回归任务 |

#### 5.2.1.1 推导过程——拓展可以不看

对于第  i  个样本，真实值 $y_{i}$ 可以看作预测值 $\hat{y}_{i}$ 加上一个噪声 $\epsilon_i$ :
$$
y_i = \hat{y}_i + \epsilon_i
$$
且噪声服从均值为 0、方差为$\sigma^2$的高斯分布：
$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$
则  y  的概率密度函数为：

$$
P(\epsilon_i) = \frac{1}{\sqrt{2\pi\sigma^2}} e^ {-\frac{\epsilon_i^2}{2\sigma^2} }
$$
由于$\epsilon_i = y_i - \hat{y}_i$，我们可以写出给定输入$x_i$下，真实值 $y_i$ 的条件概率密度：
$$
P(y_i | \hat{y}_i) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{ -\frac{(y_i - \hat{y}_i)^2}{2\sigma^2} }
$$
假设所有样本独立同分布，整组数据同时出现的概率（似然函数）为：

$$
L(\theta) = \prod_{i=1}^{n} P(y_i | \hat{y}_i) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} e^{ -\frac{(y_i - \hat{y}_i)^2}{2\sigma^2} }
$$
对似然函数取自然对数，将连乘转化为连加：
$$
\begin{aligned} \log L(\theta) &= \sum_{i=1}^{n} \left[ \log\left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) - \frac{(y_i - \hat{y}_i)^2}{2\sigma^2} \right] \\ &= -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \end{aligned}
$$
我们的目标是找到模型参数$\theta$，使得对数似然$\log L(\theta)$最大。

观察上式：

第一项$-\frac{n}{2} \log(2\pi\sigma^2)$是常数（与模型参数$\theta$无关，只与数据方差有关)

第二项  $-\frac{1}{2\sigma^2} \sum (y_i - \hat{y}_i)^2$ 中，$\frac{1}{2\sigma^2} > 0$ 是常数

因此，**最大化对数似然**等价于$\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

为了得到“平均”形式的损失，再除以样本数  $n$ （不改变最优解的位置）：
$$
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \text{MSE}
$$

#### 5.2.1.2 基本代码实现

在PyTorch中通过nn.MSELoss()实现：

```python
import torch
from torch import nn


def test04():
    # 1 设置真实值和预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    # 2 实例MSE损失对象
    loss = nn.MSELoss()
    # 3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('myloss:', my_loss)
```

### 5.2.2 平均绝对误差MAE

**mean absolute loss(MAE)**也被称为L1 Loss，是以**绝对误差**作为距离 公式如下：
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$


|   **符号**    |     **含义**      |
| :-----------: | :---------------: |
|      $n$      |     样本总数      |
|    $y_{i}$    | 第i个样本的真实值 |
| $\hat{y}_{i}$ | 第i个样本的预测值 |

优点

|       优点       |                  说明                  |
| :--------------: | :------------------------------------: |
| **对异常值鲁棒** |    误差线性增长，不放大异常值的影响    |
|   **解释性强**   | 单位与原始数据一致，可直接向业务方解释 |
|  **无假设约束**  |         不要求误差服从正态分布         |

缺点

|         缺点         |                             说明                             |
| :------------------: | :----------------------------------------------------------: |
|  **在 0 点不可导**   | \|x\| 在 ( x=0 ) 处导数不存在，影响梯度优化（实际常用次梯度或平滑近似） |
|     **梯度恒定**     | 无论误差大小，梯度恒为 ±1（除 0 点外），可能导致在最优解附近震荡 |
| **对极小误差不敏感** |  不像 MSE 在误差趋近 0 时梯度也趋近 0，MAE 无法微调细小误差  |

#### 5.2.2.1 推导过程——拓展 可以不看

对于第  i  个样本，真实值 $y_{i}$ 可以看作预测值 $\hat{y}_{i}$ 加上一个噪声 $\epsilon_i$ :
$$
y_i = \hat{y}_i + \epsilon_i
$$
我们假设噪声 $\epsilon_i$ 服从**均值为 0 的拉普拉斯分布**，其概率密度函数为：
$$
P(\epsilon) = \frac{1}{2b} e^{ -\frac{|\epsilon|}{b} }
$$
其中  b > 0  是尺度参数，控制分布的“宽度”（类似于高斯分布中的标准差）

代入 $\epsilon_i = y_i - \hat{y}_i$ ，得到给定输入 $x_i$ 下，真实值  $y_i$ 的条件概率密度：
$$
P(y_i | \hat{y}_i) = \frac{1}{2b} e^{ -\frac{|y_i - \hat{y_i}|}{b} }
$$
对于  n  个独立同分布的样本，**似然函数（联合概率）**为所有样本条件概率的乘积：
$$
L(\theta) = \prod_{i=1}^{n} P(y_i | \hat{y}_i) = \prod_{i=1}^{n} \frac{1}{2b} e^{ -\frac{|y_i - \hat{y}_i|}{b} }
$$
对似然函数取自然对数，将连乘转化为连加：

$$
\begin{aligned} \log L(\theta) &= \sum_{i=1}^{n} \left[ \log\left( \frac{1}{2b} \right) - \frac{|y_i - \hat{y}_i|}{b} \right] \\ &= n \log\left( \frac{1}{2b} \right) - \frac{1}{b} \sum_{i=1}^{n} |y_i - \hat{y}_i| \end{aligned}
$$
我们的目标是最大化对数似然$\log L(\theta)$

观察上式：

第一项  $n \log(\frac{1}{2b})$  是常数（与模型参数  $\theta$  无关)

第二项  $- \frac{1}{b} \sum_{i=1}^{n} |y_i - \hat{y}_i|$  中， $\frac{1}{b} > 0$ 是常数

因此，**最大化对数似然**等价于**最小化**$\sum_{i=1}^{n} |y_i - \hat{y}_i|$

为了得到“平均”形式的损失，再除以样本数 n （不改变最优解的位置）：
$$
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| = \text{MAE}
$$

#### 5.2.2.2 基本代码实现

在PyTorch中使用nn.L1Loss()实现

```python
import torch
from torch import nn


# 计算inputs与target之差的绝对值
def test03():
    # 1 设置真实值和预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    # 2 实例MAE损失对象
    loss = nn.L1Loss()
    # 3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('loss:', my_loss)
```

### 5.2.3 Smooth L1损失函数

**Smooth L1 损失** 是一种结合了 **MSE（L2 Loss）** 和 **MAE（L1 Loss）** 优点的损失函数，最早由

Ross Girshick 在 **Fast R-CNN** 论文中提出，主要用于目标检测中的边界框回归（Bounding Box Regression）。

其核心思想是：**在误差较小时表现得像 MSE（平滑可导、收敛精细），在误差较大时表现得像** **MAE（梯度稳定、对异常值鲁棒）。**

设误差为  $e = y - \hat{y}$，Smooth L1 损失定义为分段函数：
$$
L_{\delta}(e) = \begin{cases} 0.5 \cdot e^2, & \text{if } |e| \le \delta \\ \delta \cdot |e| - 0.5 \cdot \delta^2, & \text{otherwise} \end{cases}
$$


|   **符号**    |                 **含义**                 |
| :-----------: | :--------------------------------------: |
|   $\delta$    | 超参数，用于控制“平滑区域”的阈值 通常取1 |
|    $y_{i}$    |            第i个样本的真实值             |
| $\hat{y}_{i}$ |            第i个样本的预测值             |

当 $\delta = 1$ ，公式为 （最常见版本)：
$$
L_{\text{smooth L1}}(e) = \begin{cases} 0.5 \cdot e^2, & \text{if } |e| < 1 \\ |e| - 0.5, & \text{otherwise} \end{cases}
$$
在 $e \in [-1, 1]$  区间：使用 **MSE（二次函数)**，光滑且导数较小

在 $e \in (-\infty, -1) \cup (1, +\infty)$ 区间：使用 **MAE（线性函数)**，梯度饱和，防止异常值主导梯度

|       对比维度       |     **MSE (L2)**      |         **MAE (L1)**         |              **Smooth L1**              |
| :------------------: | :-------------------: | :--------------------------: | :-------------------------------------: |
|     **函数形式**     |         $e^2$         |           $\|e\|$            |                   $e$                   |
| **对异常值的敏感度** | **极高**（平方放大）  |          低（线性）          |      **低**（大误差时退化为线性）       |
|    **0 点可导性**    |   ✅ 可导（导数为0）   | ❌ **不可导**（导数为±1跳变） |          ✅ **可导**（导数为0）          |
|  **大误差时的梯度**  |   $2e$ （无限增大）   |        $\pm 1$（恒定)        | $\pm \delta$（恒定为常数，不放大异常值) |
|  **小误差时的梯度**  | $2e \to 0$（精细微调) |    $\pm 1$（不变，易震荡)    |          $e \to 0$（精细微调)           |
|     **计算开销**     |          低           |              低              |         中等（需判断分段条件）          |
|    **最优解偏移**    |     向异常值偏移      |          偏向中位数          |       平衡（受异常值影响小于MSE）       |

#### 5.2.3.1 基础代码实现

在PyTorch中使用nn.SmoothL1Loss()计算该损失

```python
import torch
from torch import nn


def test05():
    # 1 设置真实值和预测值
    y_true = torch.tensor([0, 3])
    y_pred = torch.tensor([0.6, 0.4], requires_grad=True)
    # 2 实例smmothL1损失对象
    loss = nn.SmoothL1Loss()
    # 3 计算损失
    my_loss = loss(y_pred, y_true).detach().numpy()
    print('loss:', my_loss)
```

# 六、网络优化方法

## 6.1 梯度下降法介绍

**梯度下降法（Gradient Descent）** 是机器学习和深度学习中最核心、最常用的一阶优化算法。

其本质是：通过不断迭代，沿着损失函数梯度下降（负梯度）的方向更新模型参数，从而找到使损

失函数值**最小**的最优参数解。

公式如下：
$$
\theta _{i + 1} = \theta _{i} - \alpha \frac{\partial J(\theta )}{\partial \theta _{i}}
$$


|                      **符号**                      |     **含义**     |
| :------------------------------------------------: | :--------------: |
|                 $\theta _{i + 1}$                  |     下一个点     |
|                   $\theta _{i}$                    |     上一个点     |
|                    $J(\theta)$                     |     损失函数     |
| $\frac{\partial J(\theta )}{\partial \theta _{i}}$ | 对损失函数求偏导 |
|                      $\alpha$                      |   学习率(步长)   |

## 6.2 反向传播——了解

**反向传播（Backpropagation）** 是训练神经网络的核心算法。它通过**微积分中的链式法则（Chain Rule）**，从网络的输出层向输入层逐层计算损失函数对每个参数的梯度（偏导数），然后将这些梯度传递给优化器（如 SGD、Adam）来更新参数。

对于复合函数  $f(g(h(x)))$ ，其导数为：
$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}
$$
反向传播就是**将这个链式法则应用到神经网络的每一层**：

设网络有  L  层，损失为  L ，则：
$$
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial a^{(L-1)}} \cdots \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}
$$
理解方法：梯度是从后往前“接力”传播的——后一层的梯度乘以本层的局部梯度，得到前一层的梯度。

设输入为  x ，权重  w ，偏置  b ，激活函数为 ![Sigmoid (\sigma)](https://latex.csdn.net/eq?Sigmoid%20%28%5Csigma%29)，损失函数为 MSE，预测值为a有：
$$
z = wx + b
$$

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}} \quad
$$

$$
L = \frac{1}{2}(y - a)^2
$$

第 1 步：损失对输出的导数
$$
\frac{\partial L}{\partial a} = \frac{\partial}{\partial a} \left[ \frac{1}{2}(y - a)^2 \right] = -(y - a) = a - y
$$
第 2 步：激活函数对线性输入的导数（局部梯度）
$$
\frac{\partial a}{\partial z} = \sigma'(z) = \sigma(z)(1 - \sigma(z)) = a(1 - a)
$$
第 3 步：损失对  z  的导数（误差信号$\delta$)
$$
\delta = \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = (a - y) \cdot a(1 - a)
$$
第 4 步：损失对权重  w  的梯度
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = \delta \cdot x
$$
第 5 步：损失对偏置  b  的梯度
$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = \delta \cdot 1 = \delta
$$
第 6 步：损失对输入  x  的梯度（用于传递给前一层）
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = \delta \cdot w
$$

## 6.3 梯度下降优化

### 6.3.1 需要优化的原因

原始的梯度下降（Vanilla GD / SGD）虽然能工作，但在实际训练中存在诸多痛点：

|           痛点           |                  表现                   |             优化算法的解决方向             |
| :----------------------: | :-------------------------------------: | :----------------------------------------: |
|      **学习率难调**      |           太大震荡，太小龟速            |   自适应学习率（AdaGrad、RMSProp、Adam）   |
| **梯度震荡（峡谷地形）** |  在陡峭方向来回摆动，平坦方向缓慢移动   | 引入**动量（Momentum）**，累积历史梯度方向 |
|  **局部极小值 / 鞍点**   | 梯度为 0 时卡住（尤其是高维空间的鞍点） |        动量提供“惯性”，冲过平坦区域        |
|       **稀疏特征**       |     某些特征出现次数极少，更新不足      |   为每个参数分配不同的学习率（AdaGrad）    |

### 6.3.2 前置知识——指数加权平均

**指数加权平均（EWMA）** ：指的是给每个数赋予不同的权重求得平均数。

移动平均数：指的是计算最近邻的 N 个数来获得平均数。

指数移动加权平均：是各数值的权重都不同，距离越远的数字对平均数计算的贡献就越小（权重较

小），距离越近则对平均数的计算贡献就越大（权重越大）。

对于时刻 t ，真实观测值为 $\theta_t$ ，EWMA 的估计值  $v_t$  通过以下公式递推：
$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \theta_t
$$


|  **符号**   |                        **含义**                         |
| :---------: | :-----------------------------------------------------: |
|   $\beta$   | 衰减系数，通常 $0 < \beta < 1$ ，控制历史数据的衰减速度 |
| $\theta_t$  |                   在 t 时刻的真实梯度                   |
|    $v_t$    |              当前时刻的指数加权平均梯度值               |
| $v_{t - 1}$ |                 历史指数加权平均梯度值                  |

初始值通常设为$v_0 = 0$

关键参数 $\beta$ ：越大越平滑（延迟越高），越小反应越灵敏（但噪声也越大）。

### 6.3.3 动量法Momentum

引入**速度（Velocity）** 变量  v ，累积历史梯度的指数加权平均，让参数更新具有“惯性”。

步骤 1：更新速度（累积梯度）
$$
v_t = \beta v_{t-1} + (1 - \beta) g_t
$$
其中  $\beta$  是动量系数（通常取 0.9），控制历史梯度的衰减速度。

步骤 2：更新参数
$$
\theta_t = \theta_{t-1} - \eta \cdot v_t
$$


|     **符号**      |           **含义**           |
| :---------------: | :--------------------------: |
|      $\beta$      |    动量系数（通常取 0.9）    |
|      $\eta$       |            学习率            |
|       $g_t$       |         当前的梯度值         |
|    $\theta _t$    |     当前时刻模型权重参数     |
| $\theta _{t - 1}$ |         历史模型参数         |
|       $v_t$       | 当前时刻的指数加权平均梯度值 |
|    $v_{t - 1}$    |    历史指数加权平均梯度值    |

为什么有效：

1、在梯度方向一致的平坦区域： v  不断累积，加速前进，使得其**很有可能跨过鞍点**

2、在梯度方向来回震荡的峡谷区域：震荡方向的梯度正负抵消， v  变小，而持续向下的方向累积增大，从而减少震荡。

#### 6.3.3.1 基本代码实现

```python
def test01():

    # 1. 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    # 2. 定义损失函数：均方误差
    loss = ((w ** 2) / 2.0).sum()

    # 3. 创建优化器：基于SGD(随机梯度下降), 加入参数momentum 就是动量法。指定参数 beta=0.9
    # 参1：(待优化的)模型参数 参2：学习率 参3：动量参数
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)

    # 4. 计算梯度值：梯度清零、反向传播、参数更新
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 反向传播
    optimizer.step() # 参数更新

    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))

    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
```

### 6.3.4 自适应梯度AdaGrad

**AdaGrad（Adaptive Gradient，自适应梯度）** 是一种为每个参数单独分配学习率的优化算法

核心思想：频繁更新的参数学习率较小，稀疏更新的参数学习率较大，从而很好地适应稀疏数据

（如 NLP、推荐系统）。

步骤 1：累积历史梯度平方和
$$
r_t = r_{t-1} + g_t \odot g_t
$$
步骤 2：计算自适应学习率并更新参数
$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{r_t + \epsilon}} \odot g_t
$$


|    **符号**    |                        **含义**                         |
| :------------: | :-----------------------------------------------------: |
|   $\theta_t$   |     第  t  步更新后的参数值（如权重  w 、偏置  b ）     |
| $\theta_{t-1}$ |             第  t - 1 步的参数值（更新前）              |
|  $J(\theta)$   |                        损失函数                         |
|     $g_t$      | 第 t 步的梯度向量， $g_t = \nabla\theta J(\theta{t-1})$ |
|     $r_t$      |            第  t  步的历史梯度平方累积和向量            |
|   $r_{t-1}$    |           第 t - 1 步的历史梯度平方累积和向量           |
|     $\eta$     |           全局学习率（超参数，通常设为 0.01）           |
|   $\epsilon$   |          极小常数，防止分母为零，通常$10^{-8}$          |
|    $\odot$     |          逐元素乘（Hadamard 积，对应位置相乘）          |
|      $t$       |              当前迭代步数（从 1 开始计数）              |

为什么自适应：

对于稀疏特征，梯度 $g_t$ 很大，但历史累积 $r_t$ 较小，学习率 $\frac{\eta}{\sqrt{r_t}}$ 较大 → **快速更新**。

对于频繁特征，历史累积 $r_t$ 较大，学习率 $\frac{\eta}{\sqrt{r_t}}$ 较小 → **缓慢微调**。

#### 6.3.4.1 优缺点

|                          优点                          |                             缺点                             |
| :----------------------------------------------------: | :----------------------------------------------------------: |
|     **无需手动调参**：每个参数自动获得合适的学习率     | **学习率单调递减**： $r_t$ 只增不减，学习率最终趋近于 0，导致**提前停止学习** |
| **稀疏数据友好**：对出现频率低的特征给予更大的更新幅度 |        **后期训练无力**：在训练后期，模型难以继续微调        |
|       **理论保证好**：对凸问题有严格的收敛性证明       |   **非凸问题表现差**：在深度学习中，过早停止学习是严重问题   |
|         适合特征维度极高且稀疏的任务（如 NLP）         |             不适合需要长期精细调整的深度学习任务             |

#### 6.3.4.2 基本代码实现

```python
def test02():
    # 1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    loss = ((w ** 2) / 2.0).sum()
    # 2 实例化优化方法：adagrad优化方法
    optimizer = torch.optim.Adagrad([w], lr=0.01)
    # 3 第1次更新 计算梯度，并对参数进行更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
```

### 6.3.5 均方根传播 RMSProp

**RMSProp（Root Mean Square Propagation，均方根传播）** ：RMSProp 是 AdaGrad 的改进

版，用指数加权移动平均来计算梯度平方的均值，它解决了 AdaGrad 学习率单调递减直至消失的

问题，在非凸优化（如深度学习）中表现优异。

步骤 1：更新梯度平方的指数移动平均：
$$
r_t = \beta r_{t-1} + (1 - \beta) (g_t \odot g_t)
$$
步骤 2：更新参数
$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{r_t + \epsilon}} \odot g_t
$$

|    **符号**    |                        **含义**                         |
| :------------: | :-----------------------------------------------------: |
|   $\theta_t$   |     第  t  步更新后的参数值（如权重  w 、偏置  b ）     |
| $\theta_{t-1}$ |             第  t - 1 步的参数值（更新前）              |
|  $J(\theta)$   |                        损失函数                         |
|     $g_t$      | 第 t 步的梯度向量， $g_t = \nabla\theta J(\theta{t-1})$ |
|    $\beta$     |            衰减系数（超参数）（通常取 0.9）             |
|     $v_t$      |    第 t 步的梯度平方的 指数加权移动平均（二阶动量）     |
|   $v_{t-1}$    |            第 t - 1 步的梯度平方指数加权平均            |
|     $\eta$     |           全局学习率（超参数，通常设为 0.01）           |
|   $\epsilon$   |         极小常数，防止分母为零，通常 $10^{-8}$          |
|    $\odot$     |          逐元素乘（Hadamard 积，对应位置相乘）          |
|      $t$       |              当前迭代步数（从 1 开始计数）              |

#### 6.3.5.2 基本代码实现

```python
import torch
import torch.nn as nn

#VS Code 终端默认编码不是 UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

#定义函数 演示 梯度下降优化算法：RMSprop算法
def test03():
    #1. 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    #2. 定义损失函数：均方误差
    loss = ((w ** 2) / 2.0).sum()
    #2 实例化优化方法：RMSprop算法，其中alpha对应beta
    optimizer = torch.optim.RMSprop([w], lr=0.01, alpha=0.9)
    #4. 计算梯度值：梯度清零、反向传播、参数更新
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 反向传播
    optimizer.step() # 参数更新
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
    #5. 第2次更新 计算梯度，并对参数进行更新
    #使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
if name == 'main':
    #test03()
```

### 6.3.6 自适应动量估计 Adam

Adam = **Momentum（一阶动量）** + **RMSProp（二阶动量）**，同时拥有两者的优点：既有惯性冲过鞍点，又能自适应调节学习率。

第一步：计算当前梯度

$$
g_t = \nabla_\theta J(\theta_{t-1})
$$
第二步：更新一阶动量（梯度均值，即 Momentum） ：
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
这是梯度的指数加权平均，代表更新方向。

第三步：更新二阶动量（梯度方差，即 RMSProp）
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t \odot g_t)
$$
这是梯度平方的指数加权平均，代表更新步长。

第四步：偏差修正（关键！防止初始时偏向 0）
 由于  $m_0 = 0$ ， $v_0 = 0$，在初始阶段， $m_t$  和 $v_t$ 会严重偏向 0（因为乘以了 $1 - \beta$  因子)。必须进行修正：
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

第五步：更新参数
$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

|      符号      |                             含义                             |
| :------------: | :----------------------------------------------------------: |
|   $\theta_t$   |       第  t  步更新后的参数值（如权重  w 、偏置  b ）        |
| $\theta_{t-1}$ |                第  t - 1 步的参数值（更新前）                |
|  $J(\theta)$   |                           损失函数                           |
|     $g_t$      |    第 t 步的梯度向量，$g_t = \nabla\theta J(\theta{t-1})$    |
|    $\beta$     |               衰减系数（超参数）（通常取 0.9）               |
|     $m_t$      |             梯度的指数加权平均，相当于 Momentum              |
|   $m_{t-1}$    |                  上一步的梯度的指数加权平均                  |
|     $v_t$      |            梯度平方的指数加权平均，相当于 RMSProp            |
|   $v_{t-1}$    |                上一步的梯度平方的指数加权平均                |
|  $\hat{m}_t$   |                     偏差修正后的一阶动量                     |
|  $\hat{v}_t$   |                     偏差修正后的二阶动量                     |
|   $\beta_1$    | 一阶动量衰减系数（超参数），控制梯度均值的衰减速度，通常取 0.9 |
|   $\beta_2$    | 二阶动量衰减系数（超参数），控制梯度平方的衰减速度，通常取 0.999 |
|     $\eta$     |             全局学习率（超参数，通常设为 0.01）              |
|   $\epsilon$   |           极小常数，防止分母为零，通常  $10^{-8}$            |
|    $\odot$     |            逐元素乘（Hadamard 积，对应位置相乘）             |
|      $t$       |                 当前迭代步数，从 1 开始计数                  |

#### 6.3.6.1 优缺点

|            优点            |                             说明                             |
| :------------------------: | :----------------------------------------------------------: |
| **结合动量和自适应学习率** | 同时拥有 Momentum（快速收敛）和 RMSProp（自适应步长）的优点  |
|        **偏差修正**        |       解决初始阶段估计偏向 0 的问题，使早期训练更稳定        |
|       **超参数鲁棒**       | 默认参数 $\eta=0.001 , \beta_1=0.9 , \beta_2=0.999$ ,在绝大多数任务中表现良好 |
|        **计算高效**        |            只需存储 $m_t$ 和 $v_t$  两个额外变量             |
|      **适用场景广泛**      |           分类、回归、GAN、NLP、CV...几乎无所不能            |



|              缺点              |                             说明                             |
| :----------------------------: | :----------------------------------------------------------: |
| **权重衰减与自适应学习率耦合** | 原版 Adam 的 L2 正则化（weight decay）与自适应学习率相互作用，效果不如预期（AdamW 解决了此问题） |
|         **可能不收敛**         | 在一些问题上，Adam 可能无法收敛到最优解（已被后续研究部分解决） |
|  **对学习率 ( \eta ) 仍敏感**  |       虽然比 SGD 鲁棒，但过大的 ( \eta ) 仍会导致发散        |
|    **泛化能力可能不如 SGD**    |    在某些 CV 任务中，SGD + Momentum 的测试精度略高于 Adam    |

#### 6.3.6.2 基础代码实现

```python
def test04():
    # 1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True)
    loss = ((w ** 2) / 2.0).sum()
    # 2 实例化优化方法：Adam算法，其中betas是指数加权的系数
    optimizer = torch.optim.Adam([w], lr=0.01, betas=[0.9, 0.99])
    # 3 第1次更新 计算梯度，并对参数进行更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))
```

## 6.4 五种方法比较

| **优化算法** |                      **优点**                       |                        **缺点**                        |                     **适用场景**                     |
| :----------: | :-------------------------------------------------: | :----------------------------------------------------: | :--------------------------------------------------: |
|   **SGD**    |                  简单、容易实现。                   |      收敛速度较慢，容易震荡，特别是在复杂问题中。      |     用于简单任务，或者当数据特征分布相对稳定时。     |
| **Momentum** |    可以加速收敛，减少震荡，特别是在高曲率区域。     | 需要手动调整动量超参数，可能会在小步长训练中过度更新。 |     用于非平稳优化问题，尤其是深度学习中的应用。     |
| **AdaGrad**  |         自适应调整学习率，适用于稀疏数据。          |    学习率会在训练过程中逐渐衰减，可能导致早期停滞。    |      适合稀疏数据，如 NLP 或推荐系统中的特征。       |
| **RMSProp**  |   解决了 AdaGrad 学习率过早衰减的问题，适应性强。   |       需要选择合适的超参数，更新可能会过于激进。       |   适用于动态问题、非平稳目标函数，如深度学习训练。   |
|   **Adam**   | 结合了 Momentum 和 RMSProp 的优点，适应性强且稳定。 |  需要调节更多的超参数，训练过程中可能会产生较大波动。  | 广泛适用于各种深度学习任务，特别是非平稳和复杂问题。 |

## 6.5 学习率优化

在训练神经网络时，一般情况下学习率都会随着训练而变化。这主要是由于，在神经网络训练的后期，如果**学习率过高，会造成loss的振荡**，但是如果**学习率减小的过慢，又会造成收敛变慢**的情况。

**学习率衰减（Learning Rate Decay / Scheduling）** 是指在训练过程中，**随着迭代次数（Epoch）的增加，逐步减小学习率**的一种训练策略。

### 6.5.1 阶梯衰减（Step Decay / MultiStep Decay）

每隔固定的 epoch 数，将学习率乘以一个衰减因子

代码如下：

```python
# 等间隔学习率衰减
def test_StepLR():

    # 1.定义变量 记录初始的 学习率lr    训练轮数epoch    训练次数iteration
    lr, epochs, iteration = 0.1, 200, 10

    # 2. 创建数据集 真实值y_true, 输入特征x, 权重参数w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)


    # 3. 创建优化器 动量法：加速模型收敛，减少震荡
    # 参1：需要优化的参数 惨2：学习率 惨3：动量参数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)

    # 4. 创建学习率衰减对象：等间隔学习率衰减
    # 参1：优化器 参2：间隔的轮数(多少轮调整一次学习率) 参3：学习率衰减倍数
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 5.创建两个列表，用于记录学习率和轮数的变化
    lr_list, epoch_list = [], []

    # 6.循环遍历训练轮数，进行训练
    for epoch in range(epochs):

        # 7.记录当前的学习率 和 轮数
        lr_list.append(scheduler_lr.get_last_lr()) # scheduler_lr.get_last_lr() 获取当前学习率
        epoch_list.append(epoch)

        # 8.循环遍历，每轮训练 所有批次
        for i in range(iteration):  # 遍历每一个batch数据

            # 9. 先计算预测值，基于损失函数计算损失
            y_pred = w * x

            # 10. 计算损失 最小二乘法
            loss = (y_pred - y_true)**2

            # 11. 计算梯度值：梯度清零、反向传播、参数更新
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 参数更新

        # 12. 更新下一个epoch的学习率
        scheduler_lr.step()

    # 13. 绘制学习率变化的曲线
    # x轴：训练轮数 y轴：每轮训练用的学习率
    plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()	
```

### 6.5.2 指定间隔学习率衰减

指定在训练第几轮进行衰减

代码如下：

```python
# 指定间隔衰减学习率
def test_MultiStepLR():
    # 1.定义变量 记录初始的 学习率lr    训练轮数epoch    训练次数iteration
    lr, epochs, iteration = 0.1, 200, 10
    
    # 2. 创建数据集 真实值y_true, 输入特征x, 权重参数w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
    
    
    # 3. 创建优化器 动量法：加速模型收敛，减少震荡
    # 参1：需要优化的参数 惨2：学习率 惨3：动量参数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
    
    # 4. 创建学习率衰减对象：指定间隔学习率衰减
    milestones = [50, 125, 160] # 指定要学习率衰减的轮数

    # 参1：优化器 参2：要衰减的轮数列表 参3：学习率衰减倍数
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=0.5)
    
    # 5.创建两个列表，用于记录学习率和轮数的变化
    lr_list, epoch_list = [], []
    
    # 6.循环遍历训练轮数，进行训练
    for epoch in range(epochs):
    
        # 7.记录当前的学习率 和 轮数
        lr_list.append(scheduler_lr.get_last_lr()) # scheduler_lr.get_last_lr() 获取当前学习率
        epoch_list.append(epoch)
    
        # 8.循环遍历，每轮训练 所有批次
        for i in range(iteration):  # 遍历每一个batch数据
    
            # 9. 先计算预测值，基于损失函数计算损失
            y_pred = w * x
    
            # 10. 计算损失 最小二乘法
            loss = (y_pred - y_true)**2
    
            # 11. 计算梯度值：梯度清零、反向传播、参数更新
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 参数更新
    
        # 12. 更新下一个epoch的学习率
        scheduler_lr.step()
    
    # 13. 绘制学习率变化的曲线
    # x轴：训练轮数 y轴：每轮训练用的学习率
    plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()
```

### 6.5.3 指数学习率衰减

每个 epoch 都将学习率乘以一个固定的衰减因子

代码如下：

```python
def test_ExponentialLR():
    # 1.定义变量 记录初始的 学习率lr    训练轮数epoch    训练次数iteration
    lr, epochs, iteration = 0.1, 200, 10
        
    # 2. 创建数据集 真实值y_true, 输入特征x, 权重参数w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
        
        
    # 3. 创建优化器 动量法：加速模型收敛，减少震荡
    # 参1：需要优化的参数 惨2：学习率 惨3：动量参数
    optimizer = optim.SGD([w], lr=lr, momentum=0.9)
        
    # 4. 创建学习率衰减对象：指数衰减学习率

    # 参1：优化器  参2: 学习率衰减倍数
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
    # 5.创建两个列表，用于记录学习率和轮数的变化
    lr_list, epoch_list = [], []
        
    # 6.循环遍历训练轮数，进行训练
    for epoch in range(epochs):
        
        # 7.记录当前的学习率 和 轮数
        lr_list.append(scheduler_lr.get_last_lr()) # scheduler_lr.get_last_lr() 获取当前学习率
        epoch_list.append(epoch)
        
        # 8.循环遍历，每轮训练 所有批次
        for i in range(iteration):  # 遍历每一个batch数据
        
            # 9. 先计算预测值，基于损失函数计算损失
            y_pred = w * x
        
            # 10. 计算损失 最小二乘法
            loss = (y_pred - y_true)**2
        
            # 11. 计算梯度值：梯度清零、反向传播、参数更新
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimizer.step() # 参数更新
        
        # 12. 更新下一个epoch的学习率
        scheduler_lr.step()
```

### 6.5.4 三者对比

|     方法     |  等间隔学习率衰减 (Step Decay)   |   指定间隔学习率衰减 (Exponential Decay)   | 指数学习率衰减 (Exponential Moving Average Decay) |
| :----------: | :------------------------------: | :----------------------------------------: | :-----------------------------------------------: |
| **衰减方式** |           固定步长衰减           |                指定步长衰减                |            平滑指数衰减，历史平均考虑             |
| **实现难度** |            简单易实现            |             相对简单，容易调整             |             需要额外历史计算，较复杂              |
| **适用场景** |    大型数据集、较为简单的任务    |         对训练平稳性要求较高的任务         |             高精度训练，避免过快收敛              |
|   **优点**   | 直观，易于调试，适用于大批量数据 |           易于调试，稳定训练过程           |        平滑且考虑历史更新，收敛稳定性较强         |
|   **缺点**   |  学习率变化较大，可能跳过最优点  | 在某些情况下可能衰减过快，导致优化提前停滞 |    超参数调节较为复杂，可能需要更多的计算资源     |

# 七、正则化方法

**正则化（Regularization）** 是指在模型训练过程中，**为防止过拟合（Overfitting）而引入的额外约束或惩罚项**，以限制模型复杂度，提升泛化能力。

## 7.1 Dropout正则化

在训练深层神经网络时，由于模型参数较多，在数据量不足的情况下，很容易过拟合。Dropout（中文翻译成随机失活）是一个简单有效的正则化方法。

![image-20260926120933093](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260926120933093.png)

原理：在训练过程中，以概率 p（通常 0.2~0.5）随机将部分神经元的输出置为 0，使其不参与前向传播和反向传播。

实际应用中，通常会在全连接层（激活函数后）之后添加Dropout层

### 7.1.1 基本代码实现

```python
import torch
import torch.nn as nn

#VS Code 终端默认编码不是 UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 定义函数 随即失活
def test():
    # 1.创建隐藏层输出结果
    t1 = torch.randint(0, 10, size=[1, 4]).float()

    # 2.进行下一层 加权求和计算 和 激活函数计算
    # 2.1 创建全连接层(充当线性层)
    # 参数1 ：输入特征维度 参数2：输出特征维度 
    linear1 = nn.Linear(4, 4)

    # 2.2 加权求和
    l1 = linear1(t1)

    # 2.3 激活函数
    output = torch.relu(l1)

    # 3. 对激活值进行随机失活处理 只有训练时才进行随机失活
    dropout = nn.Dropout(p=0.4) # p=0.4 表示有40%的概率被失活

    # 4. 具体的随机失活动作
    d1 = dropout(output) # d1表示失活后的输出结果

if __name__ == '__main__':
    test()
```

## 7.2 批量归一化 Batch Normalization

原理：对网络每一层的输出进行标准化（均值为0，方差为1），再对数据重构（缩放+平移）

从而加速训练并提高泛化能力。公式如下：
$$
f(\mathbf{x}) = \lambda \cdot \frac{\mathbf{x} - \text{E}(\mathbf{x})}{\sqrt{\text{Var}(\mathbf{x}) + \epsilon}} + \beta
$$
λ 和 β 是可学习的参数，它相当于对标准化后的值做了一个**线性变换**，**λ 为系数，β 为偏置；**

$\epsilon$是一个很小的常数，通常指为 1e-5，避免分母为 0

E(x) 表示变量的均值；Var(x) 表示变量的方差；

![image-20260926121119231](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260926121119231.png)

批量归一化层在计算机视觉领域使用较多

### 7.2.1 基本代码实现

```python
import torch
import torch.nn as nn

"""
BatchNorm1d：主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为 (N, num_features) 的张量作为输入。
BatchNorm2d：主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为 (N, C, H, W) 的张量作为输入。
BatchNorm3d：主要用于三维卷积神经网络 (3D CNN)，处理三维数据，例如视频或医学图像。它接收形状为 (N, C, D, H, W) 的张量作为输入。
"""
    
def tes01():
    # 1. 创建图像样本数据   假设是经过卷积层(Conv2d)处理后的特征图
    # (N, C, H, W): 一张图, 两个通道, 每个通道 3行 4列
    # 可以创建1个样本, 图像的BN是对每个通道的特征图(行列数据)进行标准化
    input_2d = torch.randn(size=(1, 2, 3, 4))
    print("input-->", input_2d)

    # 2. 创建批量归一化层(BN层)

    # num_features：输入特征数 = 图片通道数  eps：噪声值(小常数) 默认值1e-5
    # momentum：动量值, 用于计算移动平均统计量    affine：使用可学习的变化参数(γ, β) 默认为True 对归一化后的数据进行 缩放和平移
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1, affine=True) 

    # 3. 进行批量归一化
    output = bn2d(input_2d)
    
    print("output-->", output)
    print(output.size())

    print(bn2d.weight)
    print(bn2d.bias)  

if __name__ == '__main__':
    tes01()
```

# 八、案例——手机价格分类

## 8.1 需求分析

小明创办了一家手机公司，他不知道如何估算手机产品的价格。为了解决这个问题，他收集了多家公司的手机销售数据。该数据为二手手机的各个性能的数据，最后根据这些性能得到4个价格区间，作为这些二手手机售出的价格区间。主要包括：

![image-20260926121146956](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260926121146956.png)

结果是4个区间，实际上属于分类问题

## 8.2 代码实现

```python
import torch    # pytorch框架，封装了张量的各种操作
from torch.utils.data import TensorDataset  # 数据集对象 数据-> tensor  -> 数据集对象 -> 数据加载器
from torch.utils.data import DataLoader     # 数据加载器
import torch.nn as nn                       # neural network 封装了神经网络的各种操作
import torch.optim as optim                 # 优化器
from sklearn.model_selection import train_test_split # 划分数据集
import matplotlib.pyplot as plt # 画图
import numpy as np # 数组操作
import pandas as pd # 数据处理
import time # 时间模块
from torchsummary import summary # 模型结构可视化

#VS Code 终端默认编码不是 UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 定义函数 构建数据集
def create_dataset():

    # 1.1 读取数据
    data = pd.read_csv('./data/train.csv')

    # 1.2 获取 x特征列 和 y标签列
    # data.iloc[:, :-1] 获取所有行，除了最后一列的所有列  data.iloc[:, -1] 获取所有行，最后一列
    x, y = data.iloc[:, :-1], data.iloc[:, -1] 

    # 1.3 把特征列转化成浮点型
    x = x.astype('float32')

    # 1.4 划分数据集
    # 参数1：特征列 参数2：标签列 参数3：测试集比例 参数4：随机种子 参数5：样本的分布(参考y的类别进行抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    # 1.5 把数据集封装成 张量数据集 数据 -> 张量tensor -> 数据集对象TensorDataSet -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))

    # 1.6 返回结果
    # x_train.shape[1] 获取特征列的个数 len(y_train.unique()) 获取标签列的类别个数
    return train_dataset, test_dataset, x_train.shape[1], len(y_train.unique())

# 2. 定义函数 构建分类网络模型
class PhonePriceModel(nn.Module):
    # 2.1. 在init魔法方法中，初始化父类成员，搭建神经网络
    def __init__(self, input_dim, output_dim):

        # 2.1.1 初始化父类成员
        super().__init__()

        # 2.1.2 搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_dim)
        
    # 2.2 定义向前传播 forward()方法
    def forward(self, x):

        # 2.2.1 隐藏层1: 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear1(x))

        # 2.2.2 隐藏层2: 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))

        # 2.2.3 输出层: 加权求和 + 激活函数(softmax)
        # 正常写法，这里不需要，因为后续会使用 多分类交叉熵损失函数，它内部会自动计算 softmax
        # x = torch.softmax(self.output(x), dim=1)
        x = self.output(x)

        return x

# 3. 定义函数 训练模型
def train(train_dataset, input_dim, output_dim):

    # 3.1 准备数据加载器
    # 参数1：数据集对象 参数2：批次大小 参数3：是否打乱数据(训练集)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3.2 构建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)

    # 3.3 定义损失函数 因为是多分类 所以使用 多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 3.4 创建优化器
    # 参数1：模型参数 参数2：学习率
    optimzer = optim.SGD(model.parameters(), lr=0.001)

    # 3.5 模型训练
    # 3.5.1 训练轮数
    epochs = 50

    # 3.5.2 每轮训练
    for epoch in range(epochs):

        # 3.5.2.1 定义变量 记录每次训练的损失 训练批次数
        total_loss, batch_num = 0.0, 0

        # 3.5.2.2 定义变量 表示训练开始时间
        strat = time.time()

        # 3.5.2.3 每轮训练 每个批次
        for x, y in train_loader:

            # 3.5.2.3.1 切换模型
            model.train() # 训练模式 model.eval() 测试模式

            # 3.5.2.3.2 模型预测
            y_pred = model(x)

            # 3.5.2.3.3 计算损失
            loss = criterion(y_pred, y)

            # 3.5.2.3.4 梯度清零, 反向传播, 更新参数
            optimzer.zero_grad() # 梯度清零
            loss.backward() # 反向传播
            optimzer.step() # 更新参数

            # 3.5.2.3.5 累计损失 累计批次数
            total_loss += loss.item() # 将本轮的 每批次(16条)的 平均损失 累积起来 即：第一批的平均损失 + 第二批的 平均损失 + ... + 最后一批的平均损失
            batch_num += 1 # 累计批次数

    # 3.5.3 训练结束 保存模型(参数)
    # 参1：模型对象的参数(权重矩阵, 偏置矩阵) 参数2：保存路径
    torch.save(model.state_dict(), './model/phone_price_model.pth')# 后缀pth、pkl、pickle均可
    print(f'训练结束，模型已保存到：./model/phone_price_model.pth')

# 4. 定义函数 测试模型
def evaluate(test_dataset, input_dim, output_dim):

    # 4.1 创建神经网络分类对象
    model = PhonePriceModel(input_dim, output_dim)

    # 4.2 加载模型参数
    model.load_state_dict(torch.load('./model/phone_price_model.pth'))

    # 4.3 创建测试集的 数据加载器的对象
    # 参数1：数据集对象 参数2：批次大小 参数3：是否打乱数据(测试集)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 4.4 定义变量 记录预测正确的样本数
    correct = 0

    # 4.5 从数据集中 获取每批次的数据
    for x, y in test_loader:

        # 4.5.1 切换模型
        model.eval() # 测试模式

        # 4.5.2 模型预测
        y_pred = model(x)

        # 4.5.3 根据加权求和 得到类别 用argmax()函数获取最大值的索引
        # 因为预测值是原始得分，是该样本对4种分类的得分，而得分最高的那个类别才是这个样本的预测值
        y_pred = torch.argmax(y_pred, dim=1) # dim = 1 表示按行处理

        # 4.5.4 统计预测正确的样本数
        correct += (y_pred == y).sum()

    # 4.6 预测结束 计算准确率
    print(f'测试结束，准确率：{correct / len(test_dataset)}')


# 5. 调用
if __name__ == '__main__':

    # 5.1 准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f'训练集 数据集对象：{train_dataset}\n测试集 数据集对象：{test_dataset}\n特征列个数：{input_dim}\n标签列类别个数：{output_dim}')

    # 5.2 构建神经网络模型
    # model = PhonePriceModel(input_dim, output_dim)

    # 5.3 计算模型参数
    # 参数1：模型对象 参数2：输入数据的形状(批次大小, 输入特征数) 即每批16条 每条20列特征
    # summary(model, input_size = (16, input_dim))

    # 5.4 模型训练
    # train(train_dataset, input_dim, output_dim)

    # 5.5 模型测试
    evaluate(test_dataset, input_dim, output_dim)
```

正确率只有53%，需要进一步优化

## 8.3 网络性能优化

我们可以通过以下方面进行调优:

1. 对输入数据进行标准化
2. 调整优化方法
3. 调整学习率
4. 增加批量归一化层
5. 增加网络层数、神经元个数
6. 增加训练轮数
7. 等等...

进行下如下调整:

1. 优化方法由 SGD 调整为 Adam
2. 学习率由 1e-3 调整为 1e-4
3. 对数据进行标准化
4. 增加网络深度, 即: 增加网络参数量

代码如下：

```python
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
from sklearn.preprocessing import StandardScaler


# 构建数据集
def create_dataset():
	# 使用pandas读取数据
	data = pd.read_csv('./data/手机价格预测.csv')
	# 特征值和目标值
	x, y = data.iloc[:, :-1], data.iloc[:, -1]
	# 类型转换：特征值，目标值
	x = x.astype(np.float32)
	y = y.astype(np.int64)
	# 数据集划分
	x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88, stratify=y)
	# 优化①:数据标准化
	transfer = StandardScaler()
	x_train = transfer.fit_transform(x_train)
	x_valid = transfer.transform(x_valid)
	# 构建数据集,转换为pytorch的形式
	train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
	valid_dataset = TensorDataset(torch.from_numpy(x_valid), torch.tensor(y_valid.values))
	# 返回结果
	return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))


# 构建网络模型
class PhonePriceModel(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(PhonePriceModel, self).__init__()
		# 优化②:增加网络深度
		# 1. 第一层: 输入为维度为 20, 输出维度为: 128
		self.linear1 = nn.Linear(input_dim, 128)
		# 2. 第二层: 输入为维度为 128, 输出维度为: 256
		self.linear2 = nn.Linear(128, 256)
		# 3. 第三层: 输入为维度为 256, 输出维度为: 512
		self.linear3 = nn.Linear(256, 512)
		# 4. 第四层: 输入为维度为 512, 输出维度为: 128
		self.linear4 = nn.Linear(512, 128)
		# 5. 输出层: 输入为维度为 128, 输出维度为: 4
		self.linear5 = nn.Linear(128, output_dim)

	def forward(self, x):
		# 前向传播过程
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))
		x = torch.relu(self.linear3(x))
		x = torch.relu(self.linear4(x))
		# 后续CrossEntropyLoss损失函数中包含softmax过程, 所以当前步骤不进行softmax操作
		output = self.linear5(x)
		# 获取数据结果
		return output


# 编写训练函数
def train(train_dataset, input_dim, class_num):
	# 固定随机数种子
	torch.manual_seed(0)
	# 初始化数据加载器
	dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
	# 初始化模型
	model = PhonePriceModel(input_dim, class_num)
	# 损失函数 CrossEntropyLoss = softmax + 损失计算
	criterion = nn.CrossEntropyLoss()
	# 优化③:使用Adam优化方法, 优化④:学习率变为1e-4
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	# 遍历每个轮次的数据
	num_epoch = 50
	for epoch_idx in range(num_epoch):
		# 训练时间
		start = time.time()
		# 计算损失
		total_loss = 0.0
		total_num = 0
		# 遍历每个batch数据进行处理
		for x, y in dataloader:
			model.train()
			output = model(x)
			# 计算损失
			loss = criterion(output, y)
			# 梯度清零
			optimizer.zero_grad()
			# 反向传播
			loss.backward()
			# 参数更新
			optimizer.step()
			# 损失计算
			total_num += len(y)
			total_loss += loss.item() * len(y)
		# 打印损失变换结果
		print('epoch: %4s loss: %.2f, time: %.2fs' %
			  (epoch_idx + 1, total_loss / total_num, time.time() - start))
	# 模型保存
	torch.save(model.state_dict(), './model/phone-price-model2.pth')


def test(valid_dataset, input_dim, class_num):
	# 加载模型和训练好的网络参数
	model = PhonePriceModel(input_dim, class_num)
	# load_state_dict:将加载的参数字典应用到模型上
	# load:加载用来保存模型参数的文件
	model.load_state_dict(torch.load('./model/phone-price-model2.pth'))
	# 构建加载器
	dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
	# 评估测试集
	correct = 0
	# 遍历测试集中的数据
	for x, y in dataloader:
		# 将其送入网络中
		# model.eval()
		output = model(x)
		# 获取预测类别结果
		y_pred = torch.argmax(output, dim=1)
		# 获取预测正确的个数
		correct += (y_pred == y).sum()
	# 求预测精度
	print('Acc: %.5f' % (correct / len(valid_dataset)))


if __name__ == '__main__':
	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
	train(train_dataset, input_dim, class_num)
	test(valid_dataset, input_dim, class_num)
```





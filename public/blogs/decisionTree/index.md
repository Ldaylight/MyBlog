#  一、决策树的介绍

**决策树：**是一种基于 **树状结构** 进行决策的机器学习算法，既可用于**分类**任务，也可用于**回归**任务。

**核心思想：**通过一系列“是/否”或“多选一”的问题，将数据不断划分，使得划分后的每个子集尽可能属于同一个类别 或者 目标值尽量接近

## **1.1 决策树的组成结构**

|           组成部分            |                        含义                        |
| :---------------------------: | :------------------------------------------------: |
|    **根节点（Root Node）**    | 树的起点，包含所有样本，选择最优特征进行第一次划分 |
| **内部节点（Internal Node）** |    中间的判断节点，每个节点代表对一个特征的测试    |
|      **分支（Branch）**       |   从一个节点指向下一个节点的路径，代表特征的取值   |
|    **叶节点（Leaf Node）**    |     树的末端，不再继续划分，输出最终的预测结果     |

## 1.2 决策树的建立过程

|      要点      |                           说明                           |
| :------------: | :------------------------------------------------------: |
|  **建树本质**  |    递归地**选择最优特征**划分数据，使子集纯度越来越高    |
|  **核心步骤**  |   特征选择 → 划分数据集 → 递归构建子树 → 停止条件判断    |
| **防止过拟合** |  通过限制树深度、最小样本数、**剪枝**等手段控制树的大小  |
|    **调参**    | 在复杂度（拟合训练集）和泛化能力（预测测试集）之间找平衡 |

## 1.3 决策树的分类

| **三种主流算法**  |                  ID3、C4.5、CART                  |
| :---------------: | :-----------------------------------------------: |
| **CART 是最常用** | sklearn 默认采用 CART，支持分类和回归，生成二叉树 |

三种决策树的特征选择方式有所不同

三种决策树适用的特征类型也不同：

|    类型    |                  说明                  |        常见算法         |
| :--------: | :------------------------------------: | :---------------------: |
| **分类树** | 输出为离散的类别标签（如是/否、猫/狗） | ID3、C4.5、CART（分类） |
| **回归树** |    输出为连续的数值（如房价、温度）    |      CART（回归）       |

# 二、特征选择方式

## 2.1 信息熵

**熵：**信息论中，代表随机变量不确定度的度量。熵的大小的含义：

|    熵值    |                含义                |
| :--------: | :--------------------------------: |
|  **熵大**  | 不确定性高，信息量大，数据更"混乱" |
|  **熵小**  | 不确定性低，信息量小，数据更"纯净" |
| **熵 = 0** |    完全确定（某个事件概率为 1）    |
| **熵最大** |     所有事件等概率（完全随机）     |

信息熵的计算公式如下：
$$
H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)
$$
在同种特征中，一共n种目标类别(标签类别)的总数，$p_i$是第i种类别的概率(占比)

## 2.2 信息增益

**信息增益（Information Gain）**：是ID3决策树中用于**选择最优划分特征**的核心指标。

**本质**：衡量一个特征对分类任务提供的信息量大小

信息增益的数学定义：特征A 对 训练数据集D 的**信息增益G(D, A)**，定义为**集合D的熵H(D)** 与 在给定条件**特征A**下的 D的熵H(D|A)之差。即 **信息增益 = 熵 - 条件熵**

数学公式如下：
$$
\text{G}(D, A) = H(D) - H(D|A)
$$
条件熵的公式如下：
$$
H(D|A)= \sum_{j=1}^{k} \frac{|D_j|}{|D|} \cdot H(D_j)
$$
k 是特征 A 的**取值个数**，$\left | D_{j} \right |$是第 j 个子集的样本数，$H(D_j)$是第 j 个子集的熵

先计算其中的$H(D_j)$：
$$
H(D_j) = -\sum_{i = 1}^{n}\frac{\left | C_{ij} \right |}{\left | D_{j} \right |}log_{2}(\frac{\left | C_{ij} \right |}{\left | D_{j} \right |})
$$
其中：n代表目标类别个数，$C_{ij}$表示在第 j 个子集中，属于第 i 类的样本数， $\frac{\left | C_{ij} \right |}{\left | D_{j} \right |}$就是在子集$D_{j}$内部，第 i 种类别的比例

则将$H(D_j)$带入到$H(D|A)$可以写成：
$$
H(D|A)=\sum_{j=1}^{k} -\frac{|D_j|}{|D|} \cdot \sum_{i = 1}^{n}\frac{\left | C_{ij} \right |}{\left | D_{j} \right |}log_{2}(\frac{\left | C_{ij} \right |}{\left | D_{j} \right |})
$$
可以消去$D_j$，结果为：
$$
H(D|A)=-\sum_{j=1}^{k} \sum_{i = 1}^{n}\frac{\left | C_{ij} \right |}{\left | D \right |}log_{2}(\frac{\left | C_{ij} \right |}{\left | D_{j} \right |})
$$
而$H(D)$可以写成：
$$
H(D) = -\sum_{i = 1}^{n}\frac{\left | C_{i} \right |}{\left | D \right |}log_{2}(\frac{\left | C_{i} \right |}{\left | D \right |})
$$
所以最后公式可以写为：
$$
\text{G}(D, A) =-\sum_{i = 1}^{n}\frac{\left | C_{i} \right |}{\left | D \right |}log_{2}(\frac{\left | C_{i} \right |}{\left | D \right |}) +\sum_{j=1}^{k} \sum_{i = 1}^{n}\frac{\left | C_{ij} \right |}{\left | D \right |}log_{2}(\frac{\left | C_{ij} \right |}{\left | D_{j} \right |})
$$

## 2.3 信息增益率

**特征熵：**类似于信息熵，信息熵是看目标类别，特征熵是看一个特征的取值

**惩罚系数：**即特征熵的倒数，1/特征熵

特征取值越多，特征熵越大，惩罚系数越小，信息增益越小

**信息增益率(比)** = 信息增益 / 特征熵，公式如下**：**
$$
GainRatio(D, a) = \frac{G(D, a)}{IV(a)}
$$
其中，**Gain_Ratio(D, a)**是信息增益率(公式编译器无法打下划线，所以上方公式没下划线)

G(D, a) 是信息增益

**IV(a)**是特征熵，计算公式类似与信息熵。公式如下：
$$
IV(a) = -\sum_{v = 1}^{n}\frac{\left | D_{v} \right |}{\left | D \right |}log_{2}(\frac{\left | D_{v} \right |}{\left | D \right |})
$$


## 2.4 基尼系数

**基尼值（Gini Index / Gini Impurity）**：衡量的是**数据集的不纯度（纯度）**。它表示从**数据集D**中随机抽取两个样本，其**类别标签不一致**的概率。
$$
\text{Gini}(D) = 1 - \sum_{i=1}^{m} p_i^2
$$
一共m个类别，![p_i](https://latex.csdn.net/eq?p_i)是第i种类别的概率。基尼值**越小**，数据D的纯度**越高**

**基尼系数（Gini index）/ 基尼增益：**用某个特征划分数据集后，基尼值下降的程度。
 公式如下：
$$
GiniIndex(D, a) =\text{Gini}(D)- \sum_{v = 1}^{V}\frac{\left | D_{v} \right |}{\left | D \right |}Gini(D_{v})
$$
**基尼指数本质**：分裂前的基尼值 - 分裂后各子集基尼值的加权平均，基尼系数**越大**，该特征的分裂效果**越好**

# 三、ID3决策树

**ID3（Iterative Dichotomiser 3）** 是决策树算法中最经典的一个版本，它使用**信息增益**作为特征选择标准，通过递归方式构建决策树。

仅支持**离散型**特征，仅支持**分类**任务

ID3的构建是一个**自上而下、分而治之**的递归过程，其核心思想是：**每一步都选择信息增益最大的特征进行划分，使数据纯度提升最快。**

构建流程总览：

1、计算每个特征的信息增益

2、选择**信息增益最大**的特征，将数据集划分成若干子集

3、使用该特征做为决策树的一个节点

4、用剩余的特征重复1~3步

# 四、C4.5决策树

**C4.5（Classifier 4.5）**是 ID3 算法的改进版本，它在 ID3 的基础上做了多项重要改进，解决了 ID3 的核心缺陷。

|            ID3 的缺陷            |                 C4.5 的改进                  |
| :------------------------------: | :------------------------------------------: |
| **偏向多取值特征**（如 ID 编号） | 改用**信息增益率(比)**，加入特征自身的惩罚项 |
|       **不能处理连续特征**       |  引入**连续特征二分法**，自动寻找最优切分点  |
|        **不能处理缺失值**        |        引入缺失值处理策略（加权划分）        |
|          **容易过拟合**          |          引入**后剪枝**，简化树结构          |

分裂信息可以被理解为：特征自身的“**信息量**”。取值越多、分布越均匀的特征，其分裂信息越大，作为分母时会把信息增益“**拉低**”，从而修正了多取值特征的天然优势。

|       属性       |                   说明                    |
| :--------------: | :---------------------------------------: |
| **特征选择标准** |     **信息增益率(比)（Gain Ratio）**      |
| **支持特征类型** | **离散型 、连续型**（支持连续特征二分法） |
|    **树结构**    |                  多叉树                   |
|     **输出**     |                 分类任务                  |
|   **改进特性**   |        ✅ 支持剪枝 ✅ 支持缺失值处理        |

## 4.1 连续特征处理

C4.5 支持连续型特征（如温度、湿度、收入等），但决策树的划分分支**需要离散取值**。

**解决方法：二分法**

步骤1：对连续特征的值进行**排序**

步骤2：取**相邻两个值的中点**作为候选切分点，取平均值

步骤3：计算每个切分点的信息增益

步骤4：选择信息增益最大的切分点进行二分

|           特点           |                             说明                             |
| :----------------------: | :----------------------------------------------------------: |
| **每个连续特征只用一次** | 一个连续特征在路径上被使用后，不再重复使用（与离散特征相同） |
|      **生成二叉树**      |          连续特征总是二分为“≤阈值”和“>阈值”两个分支          |
|     **计算成本较高**     |            需要对每个连续特征排序并遍历所有切分点            |

# 五、CART决策树

**CART（Classification and Regression Trees，分类与回归树）**是决策树算法中的集大成者。

|       属性       |                说明                |
| :--------------: | :--------------------------------: |
| **特征选择标准** |     **基尼系数（Gini Index）**     |
| **支持特征类型** |          离散型 + 连续型           |
|    **树结构**    | **二叉树**（每个节点最多两个分支） |
|     **输出**     |            分类 + 回归             |
|   **重要特性**   |  二叉树结构、支持剪枝、处理缺失值  |

回归树使用平方误差最小化策略，分类生成树用基尼指数最小化策略

## 5.1 CART分类树

特征选择：使用**基尼系数**，优先选择基尼值小的特征做为节点

### 5.1.1 连续型特征处理

步骤1：将连续特征的所有取值排序
 步骤2：取**相邻两个值的平均值**作为候选切分点
 步骤3：对每个候选切分点，将数据二分
 步骤4：计算每个切分点的基尼系数
 步骤5：选择基尼系数最大的切分点作为最优切分点

### 5.1.2 案例：泰坦尼克号生存预测的案例

#### 5.1.2.1 数据说明

![image-20260925124416920](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925124416920.png)

![image-20260925124433760](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925124433760.png)

这里只选择使用Pclass、Age、Sex特征，标签列为Survived

#### 5.1.2.2 代码实现

代码如下：

```python
import pandas as pd
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.tree import DecisionTreeClassifier # 决策树分类器
from sklearn.metrics import classification_report # 分类报告
import matplotlib.pyplot as plt # 可视化
from sklearn.tree import plot_tree # 绘制决策树

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 加载数据集
data = pd.read_csv('./DecisionTree/data/train.csv')
# data.info()
# print(data.head())

# 2. 数据预处理
# 2.1 提取特征和标签
x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

# 2.2 发现Age列有缺失，使用平均值来填充
x.loc[:, 'Age'] = x['Age'].fillna(x['Age'].mean())
#print(x.info())

# 2.3 将Sex列 进行one-hot编码
x = pd.get_dummies(x, columns=['Sex'])

# 2.4 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3. 特征工程

# 4. 模型训练
# 这里是CART模型, ID3和C4.5模型需要手动实现
# 默认使用基尼系数 ，max_depth = 10 表示树的最大深度为10
estimator = DecisionTreeClassifier(max_depth=10)
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pre = estimator.predict(x_test)
print(f'预测结果：{y_pre}')

# 6. 模型评估
print(f'分类评估报告：\n {classification_report(y_test, y_pre)}')

# 7. 绘制决策图
plt.figure(figsize=(30, 20)) # 设置画布大小，最后放大会看不清，想看清可以设置大一点
# 参数1: 模型对象 参数2: 是否填充颜色 参数3: 树的最大深度
plot_tree(estimator, filled=True, max_depth=10) # 绘制决策树
plt.savefig('./DecisionTree/data/decision_tree.png') # 保存决策树图片
plt.show() 
```

## 5.2 CART回归树

CART回归树预测输出的是一个连续值。

回归树的完整构建流程：

| 步骤 |                     内容                     |                           计算方式                           |
| :--: | :------------------------------------------: | :----------------------------------------------------------: |
|  1   |              计算当前节点的 MSE              | ![MSE = \frac{1}{n}\sum_{i = 1}^{n}(y_{pred} - y_{true})^{2}](https://latex.csdn.net/eq?MSE%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%20%3D%201%7D%5E%7Bn%7D%28y_%7Bpred%7D%20-%20y_%7Btrue%7D%29%5E%7B2%7D) |
|  2   | 对每个特征（对连续特征进行处理，遍历切分点） |                       计算分裂后的 MSE                       |
|  3   |      选择使 MSE 下降最多的特征和切分点       |                          `计算ΔMSE`                          |
|  4   |        按最优切分点将数据集分为两部分        |                 左子集 ≤ 阈值，右子集 > 阈值                 |
|  5   |                递归构建子节点                |                         重复步骤 1~4                         |
|  6   |                  叶节点输出                  |                   **该节点所有样本的均值**                   |

### 5.2.1 连续型特征处理

步骤1：将连续特征的所有取值排序
 步骤2：取**相邻两个值的平均值**作为候选切分点
 步骤3：对每个候选切分点，将数据二分
 步骤4：计算每个切分点的基尼系数
 步骤5：选择基尼系数最大的切分点作为最优切分点

### 5.2.2 划分标准

CART回归树使用**均方误差MSE**来划分，计算公式如下，其中$$y_{pred}$$是预测值，$y_{true}$是样本值，n为样本数量。
$$
MSE = \frac{1}{n}\sum_{i = 1}^{n}(y_{pred} - y_{true})^{2}
$$
MSE 越大 → 节点内样本值越分散（越混乱）

### 5.2.3 案例

线性回归 和 CART回归决策树 对比

#### 5.2.3.1 数据说明

![image-20260925124622903](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925124622903.png)

#### 5.2.3.2 代码实现

代码如下：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # 可视化
from sklearn.tree import DecisionTreeRegressor # 回归决策树
from sklearn.linear_model import LinearRegression # 线性回归


#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字


# 1. 准备数据
x_train = np.array(list(range(1, 11))).reshape(-1, 1)
y_train = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9.1, 9.3])


# 2. 数据预处理 该案例不需要
# 3. 特征工程 该案例不需要

# 4. 模型训练
# 4.1 分别创建 线性回归模型 和 回归决策树模型
estimator1 = LinearRegression()
estimator2 = DecisionTreeRegressor(max_depth=1) # max_depth=1 表示树的最大深度为1
estimator3 = DecisionTreeRegressor(max_depth=3) # max_depth=3 表示树的最大深度为3

# 4.2 模型训练
estimator1.fit(x_train, y_train)
estimator2.fit(x_train, y_train)
estimator3.fit(x_train, y_train)

# 5. 模型预测
# 5.1 准备测试集的 特征数据 生成0-10的0.1间隔的数组
x_test = np.arange(0, 10, 0.1).reshape(-1, 1)

# 5.2 分别预测
y_pred1 = estimator1.predict(x_test)
y_pred2 = estimator2.predict(x_test)
y_pred3 = estimator3.predict(x_test)

# 6. 模型评估 略

# 7. 绘图
# 7.1 绘制真实值的散点图
plt.scatter(x_train, y_train, c='gray')
# 7.2 绘制线性回归模型的预测曲线
plt.plot(x_test, y_pred1, c='red', label='LinearRegression')

# 7.3 绘制回归决策树模型的预测曲线
plt.plot(x_test, y_pred2, c='blue', label='max_depth=1')
plt.plot(x_test, y_pred3, c='green', label='max_depth=3')

# 7.4 添加图例
plt.legend()

# 7.5 设置x轴 y轴 标题
plt.xlabel('data')
plt.ylabel('target')
plt.title('LinearRegression vs DecisionTreeRegressor')

plt.show()
```

结果如图：

![image-20260925124700006](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925124700006.png)

由此可知，决策树可以做回归，但是容易出现过拟合现象，一般用来做分类

# 六、决策树的剪枝

剪枝的作用：防止决策树过拟合的一种正则化方法，提高模型泛化能力

剪枝：就是**剪掉一些不必要的分支**，将一些子树的节点全部删掉，用叶子节点来替换，用更简单的树来预测。

## 6.1 剪枝类型

|          剪枝类型          |    时机    |                             方法                             |                        优点                        |                             缺点                             |
| :------------------------: | :--------: | :----------------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------------------: |
| **预剪枝（Pre-Pruning）**  | 建树过程中 | 对每个节点**划分前**进行估计，若划分不能带来决策树泛化能力提升，停止划分并标记为叶节点 |      很多分支没有展开，速度快开销小，节约资源      | 可能当前划分不能显著提升，但是后续划分可以显著提高。可能欠拟合 |
| **后剪枝（Post-Pruning）** | 建树完成后 | 自底向上考察非叶节点，如果去掉该子树能带来泛化提升，将该子树替换成叶节点 | 保留更多分支，泛化性能往往优于预剪枝。欠拟合风险小 |                  训练时间开销大、计算开销大                  |
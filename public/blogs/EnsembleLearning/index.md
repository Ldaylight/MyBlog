# 一、集成学习思想

**集成学习（Ensemble Learning）：**是机器学习中的一种思想，是一种通过多个模型组合形成一个精度更高的模型。训练时，用训练集依次训练出这些基学习器，预测时进行联合预测。

参与组合的模型叫做 **基学习器(弱学习器)**

集成学习之所以有效，是因为它能够利用多个模型之间的差异来抵消各自的偏差和误差，从而提升整体性能。

![image-20260925125145284](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125145284.png)

## 1.1 集成学习分类

主要分为Bagging思想、Boosting思想

Bagging思想：随机森林

Boosting思想：Adaboost、GBDT、XGBoost、LightGBM

### 1.1.1 Bagging思想

**Bagging（Bootstrap Aggregating，自助聚合）** ：是一种**并行**式的集成方法，通过**有放回采样**（bootstrap抽样）构建多个独立的训练子集，分别训练基学习器后综合结果。

![image-20260925125258565](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125258565.png)

#### 1.1.1.1 工作原理

“并行训练，投票（或平均）决策”：

步骤1：从原始训练集中**有放回**地抽取 N 个样本（Bootstrap采样），形成不同的训练子集

步骤2：用每个子集训练一个基学习器（通常为决策树），基学习器可以**并行训练**

步骤3：预测时，分类问题采用投票法，回归问题采用平均值法

#### 1.1.1.2 特点

|   特点   |                   说明                   |
| :------: | :--------------------------------------: |
| 训练方式 |       K 个基学习器可以**并行生成**       |
| 样本权重 |      每个训练样本的权重相等（1/N）       |
| 模型权重 |      每个基学习器的权重相等（1/K）       |
| 核心优势 | 通过模型平均**降低方差**，提高预测稳定性 |



### 1.1.2 Boosting思想

**Boosting（提升方法）：** 是一种**串行**的集成方法，通过迭代训练一系列基学习器，每个新模型都**专注于修正前一个模型的错误**，逐步提升整体性能

![image-20260925125352497](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125352497.png)

#### 1.1.2.1 工作原理

“知错能改，查漏补缺”

第1轮：用初始权重训练基学习器1

第2轮：提高被学习器1分错样本的权重，训练学习器2

第3轮：提高被学习器2分错样本的权重，训练学习器3

............

最终：组合所有弱学习器，性能好的权重大

### 1.1.3 二者比较

| 对比维度 |       Bagging        |          Boosting          |
| :------: | :------------------: | :------------------------: |
| 训练方式 |       **并行**       |   **串行**（有先后顺序）   |
| 样本权重 |   所有样本权重相等   | 动态调整，错误样本权重增大 |
| 模型权重 | 所有基学习器权重相等 |     性能好的模型权重大     |
| 主要作用 |     **降低方差**     |        **降低偏差**        |

4种算法比较：

|     算法     | 所属流派 |  基学习器  |              核心机制               |      主要作用       |
| :----------: | :------: | :--------: | :---------------------------------: | :-----------------: |
| **随机森林** | Bagging  |   决策树   |         双重随机采样 + 投票         |      降低方差       |
| **AdaBoost** | Boosting |   可任意   |         自适应调整样本权重          |      降低偏差       |
|   **GBDT**   | Boosting | CART回归树 |         拟合负梯度（残差）          |      降低偏差       |
| **XGBoost**  | Boosting | CART回归树 | GBDT + 二阶优化 + 正则化 + 工程优化 | 降低偏差 + 防过拟合 |



# 二、随机森林算法

**随机森林：**是基于**Bagging思想**实现的一种集成学习算法，采用**决策树**模型作为每一个基学习器

## 2.1 数据选择

在决策树的基础上引入了**双重随机性：**

|    随机性    |                           具体做法                           |               作用               |
| :----------: | :----------------------------------------------------------: | :------------------------------: |
| **数据随机** |          从原始数据中有放回抽样，构造不同的训练子集          |          增加模型多样性          |
| **特征随机** | 每次分裂时，只从全部特征中随机选择 n 个特征（n << N）作为候选 | 进一步增加多样性，避免强特征主导 |

可以简单理解为 行(样本)随机、列(特征)随机

## 2.2 构建过程

1、从原始训练集中有放回地抽取 N 个样本，形成一棵树的训练集

2、在每个节点分裂时，随机选择 m 个特征

3、在这 m 个特征中选择最佳分裂特征

4、每棵树完全生长，不进行剪枝

5、重复上述过程构建多棵树（默认用CART决策树）

6、预测时：分类用多数投票，回归用平均值

## 2.3 优缺点

|             优点             |               缺点               |
| :--------------------------: | :------------------------------: |
|   可处理高维数据，无需降维   | 在某些噪音较大的问题上可能过拟合 |
|       可评估特征重要性       |    对取值较多的特征会产生偏向    |
|     训练速度快，易并行化     |                —                 |
|         能处理缺失值         |                —                 |
| 不易过拟合（相比单棵决策树） |                —                 |

## 2.4 随机森林算法API

![image-20260925125440487](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125440487.png)



![image-20260925125454082](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125454082.png)



## 2.5 案例——泰坦尼克号生存预测的案例

### 2.5.1 数据分析

![image-20260925125523606](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125523606.png)



![image-20260925125545177](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125545177.png)

这里只选择使用Pclass、Age、Sex特征，标签列为Survived

### 2.5.2 代码实现

代码如下：

```python
import pandas as pd
from sklearn.model_selection import train_test_split    # 划分训练集和测试集
from sklearn.tree import DecisionTreeClassifier         # 决策树
from sklearn.ensemble import RandomForestClassifier     # 随机森林
from sklearn.model_selection import GridSearchCV        # 网格搜索

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 加载数据集
data = pd.read_csv('./EnsembleLearning/data/train.csv')

# 2. 数据的预处理
# 2.1 提取特征和标签
x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

# 2.2 发现Age列有缺失，使用平均值来填充
x.loc[:, 'Age'] = x['Age'].fillna(x['Age'].mean())

# 2.3 将Sex列 进行one-hot编码
x = pd.get_dummies(x, columns=['Sex'])

# 2.4 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3. 特征工程

# 4. 模型训练
# 4.1 创建 随机森林
estimator = RandomForestClassifier() #默认参数 n_estimators=100 代表森林中有100棵树 max_depth=None 代表树的深度没有限制

# 4.2 参数准备
params = {'n_estimators': [30, 50, 60, 90, 110], 'max_depth': [2, 3, 5, 7]}

# 4.3 创建网格搜索对象 结合 交叉验证
gs_estimator = GridSearchCV(estimator, param_grid=params, cv=2) #cv=2 代表交叉验证的折数

# 4.4 训练模型
gs_estimator.fit(x_train, y_train)

# 5. 模型预测
y_pred = gs_estimator.predict(x_test)

# 6. 模型评估
print(f'随机森林模型的准确率为：{gs_estimator.score(x_test, y_test)}')
```

# 三、Adaboost算法

**AdaBoost（Adaptive Boosting，自适应提升）** 是基于**Boosting思想**实现的一种集成学习算法，采用**决策树**模型作为每一个基学习器。通过逐步提高 **被前一步分类错误的样本的权重** 来训练一个强分类器

通过**双重加权**实现自适应提升

|   加权对象   |                             策略                             |              效果              |
| :----------: | :----------------------------------------------------------: | :----------------------------: |
| **样本权重** | 被**错误**分类的样本权重**增大**，**正确**分类的权重**减小** |   后续模型重点关注难分类样本   |
| **模型权重** |            分类误差小的模型权重大，误差大的权重小            | 优秀模型在最终决策中占主导地位 |

## 3.1 Adaboost算法过程

### 3.1.1 初始化

初始化 ：假设有n个样本，初始化时各个样本的**样本权重w均**相等，均为 ( 1/n )。

### 3.1.2 第一轮

如果特征是**连续型**，开始前需要根据特征**先排序，**取**相邻**特征的**平均值**作为分裂点**。**

依次带入分裂点求错误率（预测错误个数/当前节点样本总数）

寻找一个**错误率最小**的分裂点，更新模型权重$\alpha _{t}$、n个样本权重（t代表的是第 t 轮）

#### 3.1.2.1 计算模型权重

**模型权重**$\alpha _{t}$计算公式为：
$$
\alpha _{t} = \frac{1}{2}ln(\frac{1 - \varepsilon _{t}}{\varepsilon _{t}})
$$
$\varepsilon _{t}$表示的是第t轮选取的**最优分裂点的错误率**。

模型权重$\alpha _{t}$**每一轮只计算一次**，使用的是选出来的**最优分裂点的错误率**。

$\alpha _{t}$≥**0**，因为选出来的最优分裂点的错误率一定≤0.5

#### 3.1.2.2 计算样本权重

先更新下一轮即t + 1轮的原始样本权重**（未归一化）：**

当预测值 = 真实值，预测对了，权重降低：
$$
\bar{w}_{(t + 1, i)} = w_{(t, i)}\cdot e^{-\alpha _{t}}
$$
预测值 ≠ 真实值，预测错了，权重增加：
$$
\bar{w}_{(t + 1, i)} = w_{(t, i)}\cdot e^{\alpha _{t}}
$$
$\bar{w}_{(t + 1, i)}$表示第i个样本第t + 1轮的**原始样本权重（未归一化）**

$Z_{t}$规范化因子（所有样本权重之和)计算公式为：
$$
Z_{t} = \sum_{i = 1}^{n}\bar{w}_{(t + 1, i)}
$$
则第t + 1轮**归一化后**的样本权重为：
$$
w_{(t + 1, i)} =\frac{\bar{w}_{(t + 1, i)} }{Z_{t}}
$$

### 3.1.3 第2~m轮

每一轮的决策树分裂点划分，是在**全部样本**上进行的。唯一的区别是：每个样本的**样本权重**被上一轮改变了。

根据新的样本权重，训练下一个个学习器。直到训练出m个弱学习器

样本权重的更新公式可以总结为：
$$
w_{(t + 1, i)} =\frac{w_{(t, i)} }{Z_{t}}e^{-\alpha _{t}}
$$

## 3.2 案例——葡萄酒分类

### 3.2.1 数据分析

![image-20260925125928657](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925125928657.png)

标签是：Class label    特征选择两列：Alcohol、Hue

由于CART决策树是二叉树，只能进行二分类，而标签的值有3个(1，2，3)，所以删掉一种（这里删掉1）

### 3.2.2 代码实现 

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder          # 标签编码器
from sklearn.model_selection import train_test_split    # 划分训练集和测试集
from sklearn.tree import DecisionTreeClassifier         # 决策树分类器
from sklearn.ensemble import AdaBoostClassifier         # AdaBoost分类器
from sklearn.metrics import accuracy_score               # 准确率

# VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 读取数据
data = pd.read_csv('./EnsembleLearning/data/wine0501.csv')

# 2. 数据预处理
# 2.1 从标签列中，过滤掉类别1
data = data[data['Class label'] != 1]
# print(data['Class label'].unique())

# 2.2 提取特征列和标签列
x = data[['Alcohol', 'Hue']] # 酒精 和 色泽
y = data['Class label']

# 2.3 使用 标签编码器，将标签列中的类别2，编码为0，类别3，编码为1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 2.4 划分训练集和测试集 stratify=y，表示按照标签列的类别比例，划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y)

# 3. 特征工程 略

# 4. 模型训练
# 4.1 创建 单一决策树分类器
estimator1 = DecisionTreeClassifier(max_depth=3)
# 4.2 训练模型
estimator1.fit(x_train, y_train)
# 4.3 预测
y_pred1 = estimator1.predict(x_test)

print(f'单一决策树分类器的准确率: {accuracy_score(y_test, y_pred1)}')

# 模型创建
# 参数1：弱分类器 参数2：弱分类器的个数 参数3：学习率
# 如果不指定 `estimator` 参数，会自动使用一个默认的弱分类器——深度为 1 的决策树桩
#AdaBoost 中加入学习率，是为了控制每轮弱分类器对最终模型的贡献程度，从而防止过拟合
estimator2 = AdaBoostClassifier(estimator=estimator1, n_estimators=200, learning_rate=0.1)
# 训练模型
estimator2.fit(x_train, y_train)
# 预测
y_pred2 = estimator2.predict(x_test)

print(f'AdaBoost分类器的准确率: {accuracy_score(y_test, y_pred2)}')
```

# 四、GBDT梯度提升树

**GBDT（Gradient Boosting Decision Tree）** 是**Boosting思想**下以 **CART 回归树**为基学习器的算法，核心思想是用**损失函数的负梯度**来拟合残差

残差 = 真实值 - 预测值

GBDT 的做法：**每一轮新模型都在拟合“前一轮模型的预测残差”**。上一轮的残差作为下一轮的真实值

## 4.1 残差和梯度的关系

假设有n个样本，第i个样本的特征用xi表示，第i个样本的真实值用$y_{i}$表示，第i个样本的预测值用$f(x_{i})$表示。$r_{i}$表示第i个样本的残差，有
$$
r_{i} = y_{i} - f(x_{i})
$$
对于**平方损失**函数，
$$
L(y_{i}, f(x_{i})) = \frac{1}{2}(y_{i} - f(x_{i}))^2
$$
这里系数 1/2 是为了求导时消去 2，方便计算。

对f(x)求偏导，有
$$
-\frac{\partial L}{\partial f} = y - f(x)
$$
因此，对于平方损失函数，**负梯度正好等于残差**

## 4.2 GBDT构建过程

给定训练集 $\begin{Bmatrix}(x_{i}, y_{i}) \end{Bmatrix}_{i = 1}^{N}$，一共N(n)个样本，x表示特征值，y是真实值

当前模型：$F_{m}(x)$（第m轮后的集成模型）$f_{m}(x_{i})$表示的是第m颗树对$x_{i}$的预测值，$\gamma$也是预测值

### 4.2.1 初始化模型

在所有样本的真实值y中，找一个常数$\gamma$作为初始预测值，使得所有样本的损失之和最小

有公式：
$$
L(y, \gamma ) = \sum_{i = 1}^{n}L(y_{i}, \gamma ) = \frac{1}{2}\sum_{i = 1}^{n}(y_{i} - \gamma )^{2}
$$
对$\gamma$求偏导，有：
$$
\frac{\partial L(y, \gamma )}{\partial \gamma } = \sum_{i = 1}^{n}(y_{i} - \gamma )
$$
令$\frac{\partial L(y, \gamma )}{\partial \gamma } = 0$，有
$$
\gamma = \frac{\sum_{i = 1}^{n}y_{i}}{n}
$$
即有：
$$
F_{0}(x_{i}) =\gamma
$$
所以初始化时，初始预测值![\gamma](https://latex.csdn.net/eq?%5Cgamma)取所有**真实值的均值**，可以使得**平方误差最小**

### 4.2.2 决策树的构建

第m棵树，计算每一个样本的伪残差，第i个样本这一轮的**普通残差**公式为：
$$
r_{i}^{(m)} =-\frac{\partial}{\partial \gamma } [\frac{1}{2}(y_{i} - F_{m - 1}(x_{i}))^2] = y_{i} - F_{m - 1}(x_{i})
$$
用残差来训练决策树，即使用数据集$S = \begin{Bmatrix}(x_{i}, r_{i}^{(m)}) \end{Bmatrix}$训练决策树。

 S为当前节点样本集合，

划分前的节点的预测值，其中$i\in S$：
$$
\bar{r}_{S} = \frac{1}{\left | S \right |}\sum r_{i}^{(m)}
$$
划分前损失， 其中$i\in S$
$$
Loss(S) = \sum (r_{i}^{(m)} - \bar{r}_{S})^2
$$
寻找最优切分点，$S_{L}$ $S_{R}$为划分后的左右子集

划分之后的损失，第一个求和$i\in S_{L}$，第二个求和$i\in S_{R}$：
$$
Loss(S_{L}, S_{R}) = \sum (r_{i}^{(m)} - \bar{r}_{S_{L}})^2 + \sum (r_{i}^{(m)} - \bar{r}_{S_{R}})^2
$$
划分的增益为：
$$
Gain = Loss(S) -Loss(S_{L}, S_{R})
$$
选取使得Gain值最大的划分点，将样本分成左右两个子节点，然后对每个子节点**递归执行上述过程**，直到满足停止条件。

如果用的平方损失，每棵树的叶节点最后结果就是当前节点残差和的平均值

### 4.2.3 决策树之间

前一棵决策树的残差，做为下一棵树的目标拟合值。

### **4.2.4 最终预测结果**

最终的预测结果，是所有m棵树的累计预测，
$$
F_{m}(x_{i}) = \sum_{j = 0}^{n}f_{j}(x_{i})
$$

## 4.3 案例——泰坦尼克号生存预测的案例

### 4.3.1 数据分析

![image-20260925130859255](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925130859255.png)



![image-20260925130929823](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925130929823.png)

这里只选择使用Pclass、Age、Sex特征，标签列为Survived

### 4.3.2 代码实现

```python
import pandas as pd
from sklearn.model_selection import train_test_split    #切分训练集和测试集
from sklearn.tree import DecisionTreeRegressor          #决策树回归
from sklearn.ensemble import GradientBoostingClassifier  #梯度提升回归
from sklearn.metrics import classification_report       #模型评估 
from sklearn.model_selection import GridSearchCV        #网格搜索

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 读取数据
data = pd.read_csv('./EnsembleLearning/data/train.csv')

# 2. 数据的预处理
# 2.1 提取特征和标签
x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

# 2.2 发现Age列有缺失，使用平均值来填充
x.loc[:, 'Age'] = x['Age'].fillna(x['Age'].mean())

# 2.3 将Sex列 进行one-hot编码
x = pd.get_dummies(x, columns=['Sex'])

# 2.4 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3. 特征工程，略

# 4. 模型训练
# 4.1 创建 GDBT 模型
estimator = GradientBoostingClassifier()

# 4.2 训练模型
estimator.fit(x_train, y_train)

# 5. 模型预测
y_pred = estimator.predict(x_test)

# 6. 模型评估
print(f'梯度提升树模型的准确率为：{estimator.score(x_test, y_test)}')

# 7. 网格搜索调参
# 7.1 定义可选参数
param_dict = {
    'n_estimators': [80, 100, 110, 130],        # 弱学习器的数量
    'learning_rate': [0.3, 0.5, 0.7, 0.9],      # 弱学习器的学习率
    'max_depth': [3, 5, 7, 9]                   # 弱学习器的最大深度
}

# 7.2 创建 GDBT 模型
estimator2 = GradientBoostingClassifier()

# 7.3 创建网格搜索对象
estimator3 = GridSearchCV(estimator2, param_dict, cv=5) #cv=5表示交叉验证的折数

# 7.3 模型训练
estimator3.fit(x_train, y_train)

# 7.4 模型评估
print(f'调参后的梯度提升树模型的准确率为：{estimator3.score(x_test, y_test)}')
print(f'调参后的梯度提升树模型的最佳参数为：{estimator3.best_params_}')
```

# 五、XGBoost

**XGBoost（eXtreme Gradient Boosting，极端梯度提升）**它在GBDT的基础上进行了大量工程和算法层面的优化，使其具备了**高效、灵活、可移植**的特点，在大数据竞赛和工业界中得到了广泛应用。

XGBoost是对GDBT的改进，在损失函数中加入了**正则化项**，用于**降低模型复杂度，**防止过拟合

XGBoost是基于打分函数的结果，决定是否分枝的

## 5.1 公式推导

### 5.1.1 目标函数的构造

目标函数公式如下：
$$
obj(\theta ) = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$
第一个关于损失函数L的求和，就是GDBT关于残差的求和，表示训练损失，衡量预测值与真实值的差异

第二个关于$\Omega$的求和，才是**正则化项**，用于降低模型复杂度**，**防止过拟合

对于正则化项，公式如下：
$$
\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|
$$

|   符号    |                       含义                       |
| :-------: | :----------------------------------------------: |
|     T     |             一棵树的**叶子节点数量**             |
|   $w_j$   |     第  j  个叶子节点的**权重**（即预测值）      |
| $\gamma$  | 控制叶子节点数量的惩罚系数（**节点切分的难度**） |
| $\lambda$ |       L2 正则化系数，控制叶子权重的平方和        |
| $\alpha$  |     L1 正则化系数，控制叶子权重的绝对值之和      |

在实践中，最常用的简化版本是只使用$\gamma$ 和$\lambda$ ，也就是：
$$
\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

### 5.1.2 第  t  轮迭代的目标函数

XGBoost 采用前向分步加法模型。

假设要计算第  t  轮迭代的树 对第 i 个样本的 预测值 为$f_t(x_{i})$，则第  t  轮的预测值为
$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$
代入目标函数，得到第  t  轮的目标函数为：
$$
\text{Obj}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \sum_{k=1}^{t - 1} \Omega(f_k)+ \Omega(f_t)
$$
在每一轮迭代中，我们只需要找到一棵新的树$f_t(x)$ ，使得当前目标函数最小化。

### 5.1.3 二阶泰勒展开近似

直接优化上述目标函数是困难的。XGBoost 的核心创新之一是利用**二阶泰勒展开**对目标函数进行近似。

泰勒展开公式为：
$$
f(x + \Delta x) = f(x) + f'(x)\cdot \Delta x + \frac{1}{2}f''(x)\cdot \Delta x^2 + ......+\frac{1}{n!}f^{(n)}(x)\cdot \Delta x^n
$$
二阶泰勒为：
$$
f(x + \Delta x) \approx f(x) + f'(x)\cdot \Delta x + \frac{1}{2}f''(x)\cdot \Delta x^2
$$
将损失函数$L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i))$在$\hat{y}_i^{(t-1)}$处做二阶泰勒展开：
$$
L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)
$$
其中：

这是一阶偏导数:
$$
g_i = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}
$$
这是二阶偏导数:
$$
h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}
$$
则目标函数可以写为：
$$
\tilde{\text{Obj}}^{(t)} \approx \sum_{i=1}^{n} \left[L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \sum_{k=1}^{t - 1} \Omega(f_k)+ \Omega(f_t)
$$
由于我们之后要求导找最小值，原函数常数项对求导结果没有影响，可以去掉

其中前  t-1  棵树的复杂度之和$\sum_{k=1}^{t - 1} \Omega(f_k)$可以视为常数，$L(y_i, \hat{y}_i^{(t-1)})$这个是真实值减去上一轮预测值，也是常数，这两个都可以去掉。

目标函数可以写为：
$$
\tilde{\text{Obj}}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$
带入$\Omega(f_t)$有：
$$
\tilde{\text{Obj}}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

|                        符号                         |             含义              |
| :-------------------------------------------------: | :---------------------------: |
| ![f_t(x_i)](https://latex.csdn.net/eq?f_t%28x_i%29) |    表示第 t 轮样本的预测值    |
|                          T                          |        叶子节点的数目         |
|                       gi, hi                        | 第i个样本的一阶导数、二阶导数 |

### 5.1.4 叶子节点归组——————更新位置

目前公式还是很难计算，尝试进一步化简。

定义映射函数 q(x) 将样本 x 映射到叶子节点，$w_{q(x)}$ 表示样本x对应所在的叶子节点的权重

一棵树可以表示为：
$$
f_t(x) = w_{q(x)}, \quad w \in \mathbb{R}^T
$$
将属于第 j 个叶子节点的所有样本归为一组：
$$
I_j = \{ i \mid q(x_i) = j \}
$$
将目标函数中的 $f_t(x_i)$ 替换为 $w_{q(x_i)}$，并按叶子节点归组，有：
$$
\sum_{i=1}^{n} g_i f_t(x_i) = \sum_{j=1}^{T} \left( \sum_{i \in I_j} g_i \right) w_j
$$

$$
\sum_{i=1}^{n} \frac{1}{2} h_i f_t^2(x_i) = \frac{1}{2} \sum_{j=1}^{T} \left( \sum_{i \in I_j} h_i \right) w_j^2
$$

所以，目标函数可以写为：
$$
\text{O}\tilde{\text{bj}}^{(t)} = \sum_{j=1}^{T} \left[ \left( \sum_{i \in I_j} g_i \right) w_j + \frac{1}{2} \left( \sum_{i \in I_j} h_i + \lambda \right) w_j^2 \right] + \gamma T
$$
对于固定的树结构  $q(x)$ ，目标函数是关于叶子节点权重 $w_j$ 的**独立二次函数**。

### 5.1.5 求解叶子节点的最优权重

对  $w_j$ 求导并令其为零：得到叶子节点 j 的最优权重：
$$
w_j^* = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}
$$

### 5.1.6 结构分数（评估树的质量）

将最优权重 $w_j^*$ 代入目标函数，得到**结构分数（Structure Score）：**
$$
\text{Obj}^* = -\frac{1}{2} \sum_{j=1}^{T} \frac{\left( \sum_{i \in I_j} g_i \right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T
$$
结构分数也叫打分函数

**结构分数**衡量了一棵树的质量：**分数越低（负值越大），表示树的结构越好**。

### 5.1.7 节点分裂的增益计算

XGBoost 采用**贪心算法**进行节点分裂。假设在某个节点，样本集合为 $I$，分裂后分为左子节点 $I_L$ 和右子节点$I_R$ 。

分裂前后的结构分数变化（**增益**）为：
$$
\text{Gain} = Obj_I - Obj_{I_L} - Obj_{I_R} = \frac{1}{2} \left[ \frac{\left( \sum_{i \in I_L} g_i \right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left( \sum_{i \in I_R} g_i \right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left( \sum_{i \in I} g_i \right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma
$$
**分裂决策**：只有当 **Gain > 0** 时，才执行分裂。否则，该节点停止分裂，成为叶节点。

## 5.2 XGBoost算法API

在sklearn机器学习库中，没有集成XGB，需要手动安装

打开命令行，输入：

```bash
pip3 install xgboost
```

![image-20260925135814034](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925135814034.png)

顺便下载jieba库下个学习可能会用到。

```bash
pip install jieba
```

XGBoost算法API常用的参数：

![image-20260925135840864](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925135840864.png)

## 5.3 案例——红酒品质分类

### **5.3.1 数据分析**

![image-20260925135900271](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925135900271.png)

标签列：quality

### 5.3.2 代码实现

```python
import joblib # 保存和加载模型
import pandas as pd
import numpy as np
import xgboost as xgb # XGBoost
from collections import Counter # 统计数据
from sklearn.model_selection import GridSearchCV, train_test_split # 划分训练集和测试集
from sklearn.metrics import classification_report # 模型分类评估报告
from sklearn.model_selection import StratifiedKFold  # 分层K折交叉验证，类似网格搜索
from sklearn.utils import class_weight # 计算类别权重

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 读取数据，对 原数据 拆分成 训练集和测试集 ，存储到csv中
def data_split():
    # 1.1 读取数据
    data = pd.read_csv('./EnsembleLearning/data/WineClassify.csv')

    # 1.2 提取特征 和 标签
    x = data.iloc[:, :-1]       # 除了最后一列之外，其他列都是特征
    y = data.iloc[:, -1] - 3    # 最后一列是标签, 标签从 3 开始，所以减去 3

    # 1.3 划分训练集和测试集
    # 参1：特征 参2：标签 参3：测试集占比 参4：随机种子 参5：参考数据集的标签分布
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23 ,stratify=y)

    # 1.4 将数据拼接到一起，写到文件中
    # 训练集
    pd.concat([x_train, y_train], axis=1).to_csv('./EnsembleLearning/data/WineClassify_train.csv', index=False)
    # 测试集
    pd.concat([x_test, y_test], axis=1).to_csv('./EnsembleLearning/data/WineClassify_test.csv', index=False)


# 2. 数据的预处理
def train_model():
    # 2.1 读取数据
    train_data = pd.read_csv('./EnsembleLearning/data/WineClassify_train.csv')
    test_data = pd.read_csv('./EnsembleLearning/data/WineClassify_test.csv')

    # 2.2 提取特征 和 标签
    x_train = train_data.iloc[:, :-1]   # 除了最后一列之外，其他列都是特征
    y_train = train_data.iloc[:, -1]    # 最后一列是标签
    x_test = test_data.iloc[:, :-1]     # 除了最后一列之外，其他列都是特征
    y_test = test_data.iloc[:, -1]      # 最后一列是标签

    # 2.3 创建模型对象
    estimator  = xgb.XGBClassifier(
        max_depth = 5,          # 树的最大深度
        n_estimators = 100,     # 树的数量
        learning_rate = 0.1,    # 学习率
        random_state = 23,      # 随机种子
        objective = 'multi:softmax', # 多分类模型
    )

    # 加入平衡权重 因为数据集不平衡
    # 参1：平衡权重 参2：标签数据(参考标签分布)
    class_weight.compute_sample_weight('balanced', y_train)

    # 2.4 模型训练
    estimator.fit(x_train, y_train)

    # 2.5 模型评估
    print(f'准确率：{estimator.score(x_test, y_test)}')

    # 2.6 保存模型
    # 后缀名也可以写.pth 都是pickle的格式
    joblib.dump(estimator, './EnsembleLearning/model/WineClassify_model.pkl')

# 3. 测试模型
def test_model():
    # 3.1 读取数据
    train_data = pd.read_csv('./EnsembleLearning/data/WineClassify_train.csv')
    test_data = pd.read_csv('./EnsembleLearning/data/WineClassify_test.csv')
    
    # 3.2 提取特征 和 标签
    x_train = train_data.iloc[:, :-1]   # 除了最后一列之外，其他列都是特征
    y_train = train_data.iloc[:, -1]    # 最后一列是标签
    x_test = test_data.iloc[:, :-1]     # 除了最后一列之外，其他列都是特征
    y_test = test_data.iloc[:, -1]      # 最后一列是标签

    # 3.3 加载模型
    estimator = joblib.load('./EnsembleLearning/model/WineClassify_model.pkl')

    # 3.4 创建网格搜索 + 交叉验证(结合分层采样数据) 寻找模型最优参数
    # 参数取值范围
    param_dict = {
        'max_depth': [2, 3, 5, 6],
        'n_estimators': [30, 50, 100, 120],
        'learning_rate': [0.2, 0.3, 1, 1.3],
    }
    # 创建分层采样 对象
    # 参1：划分几折 参2：是否打乱数据 参3：随机种子
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
    # 创建网格搜索对象
    # 参1：模型 参2：参数组合 参3：交叉验证对象
    gs_estimator = GridSearchCV(estimator, param_grid=param_dict, cv=skf)

    # 3.5 模型训练
    gs_estimator.fit(x_train, y_train)

    # 3.6 模型预测
    y_pred = gs_estimator.predict(x_test)

    # 3.7 模型评估
    print(f'最优估计器对象组合: {gs_estimator.best_estimator_}')
    print(f'最优评分: {gs_estimator.best_score_}')
    print(f'准确率：{gs_estimator.score(x_test, y_test)}')
    

if __name__ == '__main__':
    
    # data_split() # 1. 读取数据，对 原数据 拆分成 训练集和测试集 ，存储到csv中
    # train_model() 
    test_model()
```


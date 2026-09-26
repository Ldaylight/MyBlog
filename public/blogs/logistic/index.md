# 一、逻辑回归简介

逻辑回归（Logistic Regression）是一种二分类算法，属于有监督学习，有特征、有标签，标签离散。

它的输出不是直接的类别标签（0 或 1），而是**样本属于某个类别的概率**。

## 1.1 应用场景

预测疾病(阴性/阳性)、银行信任贷款(放贷/不放贷)、感情分析(正面/负面)、预测广告点击率(点击/不点击)

## 1.2 数学知识补充

sigmoid函数，也可以叫激活函数，数学公式如下：
$$
f(x) = \frac{1}{1 + e^{-x}}
$$
![image-20260925122326223](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925122326223.png)

是单增函数，拐点在x = 0，y = 0.5位置

作用：把值域的(-∞，+∞)映射到(0, 1)区间



条件概率：在事件A发生的条件下，事件B发生的概率为 P(B|A) = A的概率 * B的概率

极大似然估计：在**已知结果**的情况下，**反推**估算最有可能导致这个结果发生的**参数**。

# 二、逻辑回归原理

把线性回归处理后的值，通过Sigmoid激活函数映射到[0, 1]区间，结合阈值位置划分正负样本

对输入特征 `x₁, x₂, ..., xₙ` 做加权求和： y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b

y 的范围是 `(-∞, +∞)`，但我们希望输出一个概率，范围必须在 `[0, 1]` 之间。

## 2.1 损失函数

引入 **Sigmoid 函数** 将 y 压缩到 `[0, 1]` 区间：
$$
p = \sigma (y) = \frac{1}{1 + e^{-y}}
$$
**Sigmoid 函数性质：**

- 当 `y → +∞` 时，`p → 1`
- 当 `y → -∞` 时，`p → 0`
- 当 `y = 0` 时，`p = 0.5`

设定一个阈值（通常为 0.5）：

- 如果 `p ≥ 0.5`，预测为类别 1
- 如果 `p < 0.5`，预测为类别 0

## 2.2 损失函数的推导

### 2.2.1 **单个样本的概率表达式**

对于第 `i` 个样本，真实标签为 `yᵢ ∈ {0, 1}`，模型预测的正类概率为 `pᵢ = σ(yᵢ)`。

模型认为该样本属于**真实类别**的概率可以统一写成：
$$
P(y_{i} | x_{i}, w) = p_{i}^{y_{i}}(1 - p_{i})^{1 - y_{i}}
$$


- 若 `yᵢ = 1`，概率为 `pᵢ`
- 若 `yᵢ = 0`，概率为 `1 - pᵢ`

### 2.2.2 似然函数

假设所有样本独立同分布，整个数据集的联合概率（似然函数）为：
$$
L(w) = \prod _{i=1}^{m}p_{i}^{y_{i}}(1 - p_{i})^{1 - y_{i}}
$$
将连乘转化成连加：
$$
\varrho (w) = logL(w) = \sum_{i=1}^{m}[y_{i}log(p_{i})+(1 - y_{i})log(1 - p_{i})]
$$
为什么可以取对数？

| **单调性不变** | 对数函数 `log(x)` 是严格单调递增的。这意味着：如果 `A > B`，那么 `log(A) > log(B)`。所以**最大化似然函数** 和 **最大化对数似然函数** 是等价的，最优解 `w` 完全一样。 |
| :------------: | :----------------------------------------------------------: |

我们想要**最大化**对数似然，等价于**最小化**它的相反数：
$$
Loss(w) = -\varrho (w) = -\sum_{i=1}^{m}[y_{i}log(p_{i})+(1 - y_{i})log(1 - p_{i})]
$$
这就是**交叉熵 损失函数（Cross-Entropy Loss）**，也称**对数损失（Log Loss)**。

然后求梯度寻找极值点。

## 2.3 损失函数的直观理解

正类负类不代表对错，而是两种分类。

根据预测概率 `p` 和阈值得到预测结果。

| 真实标签 `y` | 预测概率 `p` | 预测结果  | 损失值 |                  直观含义                  |
| :----------: | :----------: | :-------: | :----: | :----------------------------------------: |
|      1       |     0.99     | 1（正类） |  0.01  |      ✅ 预测正确且非常自信，几乎无损失      |
|      1       |     0.90     | 1（正类） |  0.11  |        ✅ 预测正确且较自信，损失很小        |
|      1       |     0.70     | 1（正类） |  0.36  |       🤔 预测正确但信心不足，损失中等       |
|      1       |     0.50     | 1（正类） |  0.69  | 😰 恰好卡在阈值上，相当于随机猜测，损失较大 |
|      1       |     0.30     | 0（负类） |  1.20  |  😱 **预测错误**，且倾向错误方向，损失很大  |
|      1       |     0.01     | 0（负类） |  4.61  |   💀 **预测完全错误**且极其自信，损失巨大   |
|      0       |     0.01     | 0（负类） |  0.01  |      ✅ 预测正确且非常自信，几乎无损失      |
|      0       |     0.10     | 0（负类） |  0.11  |        ✅ 预测正确且较自信，损失很小        |
|      0       |     0.30     | 0（负类） |  0.36  |       🤔 预测正确但信心不足，损失中等       |
|      0       |     0.50     | 1（正类） |  0.69  | 😰 恰好卡在阈值上，相当于随机猜测，损失较大 |
|      0       |     0.70     | 1（正类） |  1.20  |  😱 **预测错误**，且倾向错误方向，损失很大  |
|      0       |     0.99     | 1（正类） |  4.61  |   💀 **预测完全错误**且极其自信，损失巨大   |

在模型训练时，损失值的**绝对大小不重要，变化的趋势才重要**。

如果损失值在**不断下降**，就说明模型正在“学习”，参数在朝着正确的方向更新。

当损失值经过很多轮训练后，下降变得非常平缓，几乎不再变化时，就说明模型已经“收敛”，可以停止训练了。

# 三、逻辑回归API和案例

## 3.1 逻辑回归API

```python
sklearn.linear_model.LogisticRegression(solver = 'liblinear', penalty = 'l2', C = 1.0)
```

solver：liblinear对小数据集训练速度更快，sag和saga对大数据集更快

​        liblinear和saga支持L1正则化，sag和saga支持L2正则化

penalty：正则化种类

C：正则化力度

默认把类别**数量少**的当作**正类**

## 3.2 案例：癌症分类预测

### 3.2.1 数据描述

![image-20260925122716132](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925122716132.png)

### 3.2.2 代码实现

代码如下：

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression # 逻辑回归模型
from sklearn.preprocessing import StandardScaler   # 标准化
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.metrics import accuracy_score # 模型评估

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1.读取数据
data = pd.read_csv('./LogisticRegression/data/breast-cancer-wisconsin.csv')
# data.info() # 打印数据集信息


# 2.数据预处理
# 2.1 由于存在缺失值，需要替换缺失值? 为np.nan      np.nan是代表缺失值
data.replace('?', np.nan, inplace=True) # 参数1：要替换的值 参数2：替换后的值 参数3：是否在原数据上修改
# 2.2 删除缺失值
data.dropna(axis=0, inplace=True) #axis = 0表示行  inplace=True 表示在原数据上修改

#data.info() # 打印数据集信息

# 3.特征工程
# 3.1 特征提取
x = data.iloc[:, 1:-1] #标号从0开始 提取特征值
y = data.iloc[:, -1] # 提取标签值 只有最后一列

print(x[:5])
print(y[:5])

# 3.2 切割训练集和测试集
# train_test_split() 参1：特征数据 参2：标签数据 参3：测试集所占比例 参4：随机种子
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 3.3 特征标准化
# 3.3.1 创建标准化对象
transfer = StandardScaler()
# 先用训练集 fit（计算均值和标准差）
transfer.fit(x_train)
# 3.3.2 对训练集和测试集进行标准化
x_train = transfer.transform(x_train)
x_test = transfer.transform(x_test)

# 4.模型训练
# 4.1 创建逻辑回归模型，在 LogisticRegression() 的内部，默认就是交叉熵损失
estimator = LogisticRegression() #默认使用L2正则化 C=1.0 solver='lbfgs' 
# 4.2 训练模型
estimator.fit(x_train, y_train)

# 5.模型预测
# 阈值也不在代码里显式出现，它被硬编码在 predict() 方法内部了，默认是 0.5
# 想要自定义阈值，必须自己用 predict_proba() 手动判断
y_pre = estimator.predict(x_test)
print(f'预测值为{y_pre}')

# 6.模型评估
# 需要通过混淆矩阵来评测 即：精确率、召回率、F1值、ROC曲线、AUC值
```

只通过准确率对模型进行评估，不能满足各种场景需要，需要通过混淆矩阵来评测

# 四、分类问题的评估

## 4.1 混淆矩阵

混淆矩阵：用一个表格的形式，把模型“在哪些样本上猜对了、在哪些样本上猜错了”以及“具体怎么错的”都清清楚楚地列出来。

T：true    F：false    P：positive    N：negative

|     真实 \ 预测     |     预测为正类（1）     |     预测为负类（0）     |
| :-----------------: | :---------------------: | :---------------------: |
| **真实为正类（1）** | **TP（真阳性）** ✅ 正确 | **FN（假阴性）** ❌ 漏报 |
| **真实为负类（0）** | **FP（假阳性）** ❌ 误报 | **TN（真阴性）** ✅正确  |

四个指标

|  缩写  |      全称      |           含义           |        通俗说法         |    是预测正确吗？    |
| :----: | :------------: | :----------------------: | :---------------------: | :------------------: |
| **TP** | True Positive  | 真实是正类，预测也是正类 |         命中了          |        ✅ 正确        |
| **TN** | True Negative  | 真实是负类，预测也是负类 |        正确放行         |        ✅ 正确        |
| **FP** | False Positive | 真实是负类，预测却是正类 | **误报** / 把好人当坏人 | ❌ 错误（第一类错误） |
| **FN** | False Negative | 真实是正类，预测却是负类 | **漏报** / 把坏人当好人 | ❌ 错误（第二类错误） |

## 4.2 精确率、召回率、F1-score

**准确率（Accuracy）会骗人，**当数据**不平衡**时，准确率会给出虚假的乐观信号。所以不能只用准确率。

|           指标            |               公式               |                含义                |
| :-----------------------: | :------------------------------: | :--------------------------------: |
|  **准确率（Accuracy）**   |    $(TP+TN) / (TP+TN+FP+FN)$     |       所有样本中，猜对的比例       |
|  **精确率（Precision）**  |     $P = \frac{TP}{TP + FP}$     | 预测为正类的所有样本中，预测准确率 |
|   **召回率（Recall）**    |     $R = \frac{TP}{TP + FN}$     | 在真实为正类中，被成功找出来的比例 |
|       **F1-score**        | $\frac{2 \cdot T\cdot P}{T + P}$ |      精确率和召回率的调和平均      |
| **特异度（Specificity）** |          $TN / (TN+FP)$          |    真实负类中，被正确排除的比例    |

**精确率 和 召回率 的关系：**通常提高一个，另一个就会下降。需要根据业务场景在两者之间做取舍。

样例代码实现：

```python
import pandas as pd
#                            混淆矩阵            精确率             召回率       F1值
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 定义数据集, 表示: 真实样本(共计10个, 6个恶性, 4个良性) => 设置: 恶性(正类), 良性(反类)
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性',   '良性', '良性', '良性', '良性']

# 2. 定义标签名
label = ['恶性', '良性']   # 正样本(正类), 负样本(反类)
df_label = ['恶性(正类)', '良性(反类)']

# 3. 定义 预测结果, 预测对了3个恶性肿瘤, 4个良性肿瘤.
y_pre = ['恶性', '恶性', '恶性', '良性', '良性', '良性',   '良性', '良性', '良性', '良性']

# 4. 把 预测结果 转换成 混淆矩阵
# 参1: 真实样本, 参2: 预测样本, 参3: 样本标签(正类, 反类)
cm = confusion_matrix(y_train, y_pre, labels=label)
print(f'混淆矩阵A: \n {cm}')

# 5. 把混淆矩阵 转换成 DataFrame.
df = pd.DataFrame(cm, index=df_label, columns=df_label)
print(f'预测结果A对应的DataFrame对象: \n {df}')
print('-' * 22)

# 9.打印预测结果 y_pre 的精确率, 召回率, F1值.
# 参1: 真实样本, 参2: 预测样本, 参3: positive label: 正类的标签
print(f'预测结果A的精确率: {precision_score(y_train, y_pre, pos_label="恶性")}')# 精确率
print(f'预测结果A的召回率: {recall_score(y_train, y_pre, pos_label="恶性")}')# 召回率
print(f'预测结果A的F1值: {f1_score(y_train, y_pre, pos_label="恶性")}')# F1值
```

## 4.3 分类评估报告

分类评估报告（Classification Report）：用一张表格同时展示每个类别的**精确率（Precision）**、**召回率（Recall）**、**F1-score** 和**支持度（Support）**。

支持度（Support）：该类别的真实样本数量

如：

![image-20260925122925837](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925122925837.png)

报告底部的三项：

下面表格的概率p是模型预测出来的，在用阈值判断之前算出来的

|             指标             |                           计算方式                           |                           适用场景                           |
| :--------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    **accuracy（准确率）**    | `(TP + TN) / (TP + TN + FP + FN)` 所有样本中，预测正确的比例 | **类别均衡**的数据集。 ⚠️ **慎用场景**：类别严重不均衡时，准确率会虚高（如 99% 负类，全猜负类也能有 99% 准确率），此时应参考其他指标。 |
|   **macro avg（宏平均）**    | 每个类别的指标独立计算后，**直接取算术平均**。 公式：`(P₁ + P₂ + ... + Pₙ) / n` 每个类别权重相同（均为 `1/n`） | **每个类别都同等重要**的场景。 例如：罕见病检测（少数类样本少，但漏诊代价极高）、多语种分类（每种语言都重要）。 能真实反映模型在**少数类**上的表现，不会被多数类“淹没”。 |
| **weighted avg（加权平均）** | 每个类别的指标按**样本数量（Support）** 加权后取平均。 公式：`(P₁×N₁ + P₂×N₂ + ... + Pₙ×Nₙ) / 总样本数` 样本越多的类别，权重越大 | **关心总体表现**，或各类别样本数量与业务重要性成正比的场景。 例如：垃圾邮件过滤（正常邮件占绝大多数，模型在正常邮件上的表现直接影响用户体验）。 ⚠️ **注意**：会**掩盖少数类的问题**，当宏平均和加权平均差距较大时，说明数据不均衡，需要警惕。 |

## 4.4 ROC曲线(了解)

**ROC 曲线（Receiver Operating Characteristic Curve，受试者工作特征曲线）：** 是一条用来衡量二分类模型区分能力的曲线。

|                  指标                   |         公式         |                          含义                          |
| :-------------------------------------: | :------------------: | :----------------------------------------------------: |
| **TPR（True Positive Rate）** 真正例率  | $\frac{TP}{TP + FN}$ | 也叫**召回率（Recall）**。正类样本中，被正确预测的比例 |
| **FPR（False Positive Rate）** 假正例率 | $\frac{FP}{FP + TN}$ |           负类样本中，被错误预测为正类的比例           |

TPR 越高越好（希望把正类都找出来），FPR 越低越好（希望不要把负类误判为正类）

ROC曲线以模型的真正例率**TPR为纵轴**y，假正利率**FPR为横轴**x，将模型在**不同阈值**下的表现画出来

### 4.4.1 ROC曲线的绘制

案例：某网站广告的点击概率，在不同阈值的情况下，绘制ROC曲线

![image-20260925123102955](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925123102955.png)



![image-20260925123116345](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925123116345.png)

## 4.5 AUC指标(了解)

**AUC（Area Under the Curve）**：就是 **ROC 曲线下方的面积**。

AUC 是一个数值，范围在 **0 到 1** 之间。AUC **越大**，模型的**区分能力越强**。

ROC曲线的优劣 可以通过曲线下方的面积AUC来衡量。

### 4.5.1 AUC 的数值含义

|  AUC 值   | 模型区分能力 |                  说明                  |
| :-------: | :----------: | :------------------------------------: |
| 0.9 ~ 1.0 |     极好     |           模型区分能力非常强           |
| 0.8 ~ 0.9 |     良好     |          模型有较好的区分能力          |
| 0.7 ~ 0.8 |     一般     |          模型有一定的区分能力          |
| 0.6 ~ 0.7 |     较差     |            模型区分能力较弱            |
|    0.5    |     无效     |     等于随机猜测，模型没有区分能力     |
|   < 0.5   |    有问题    | 模型预测比随机猜测还差（可能标签反了） |

# 五、one-hot编码处理

**One-Hot 编码：**是一种将**类别型数据**（如颜色、城市、性别）转换为**数值型数据**的方法，便于机器学习模型处理。

- **核心思想**：每个类别创建一个独立的特征列，用 `0` 和 `1` 表示是否存在
- **名称由来**：每一行数据中，**只有一个位置是 1（热）**，其余都是 0（冷）

原始数据：

| 颜色 |
| ---- |
| 红色 |
| 绿色 |
| 蓝色 |
| 红色 |

One-Hot 编码后

| 颜色_红色 | 颜色_绿色 | 颜色_蓝色 |
| :-------: | :-------: | :-------: |
|     1     |     0     |     0     |
|     0     |     1     |     0     |
|     0     |     0     |     1     |
|     1     |     0     |     0     |

为什么使用One-Hot：

|         问题         |                             说明                             |
| :------------------: | :----------------------------------------------------------: |
| **避免大小关系误导** | 如果用 `红=1, 绿=2, 蓝=3`，模型会误认为 `蓝 > 绿 > 红`，产生错误推理 |
|     **类别平等**     |         One-Hot 让所有类别在数学上地位平等，互不干扰         |
|     **模型兼容**     |   大多数机器学习算法（如逻辑回归、神经网络）只接受数值输入   |

# 六、电信客户流失预测案例

## 6.1 数据描述

![image-20260925123205374](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925123205374.png)

## 6.2 代码实现

需要对Churn和gender列进行one-hot编码处理

完整代码如下：

```python
import numpy as np
import pandas as pd
import seaborn as sns #数据可视化
import matplotlib.pyplot as plt #数据可视化
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.linear_model import LogisticRegression #逻辑回归
#                               准确率          精确率          召回率          F1值          分类评估报告
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, classification_report 

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

#1. 定义函数，实现数据预处理
def data_preprocess():
    #1.1. 读取csv文件，获取到df对象
    data = pd.read_csv('./LogisticRegression/data/churn.csv')
    # data.info() #查看数据集信息

    #1.2. 由于csv文件中的Churn列和gender列是字符串类型，需要进行one-hot编码(热编码处理)
    data = pd.get_dummies(data,columns=['Churn','gender'])
    # data.info() #查看数据集信息

    #1.3. 删除one-hot编码处理后冗余的列
    data.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True) # axis=1表示删除列，inplace=True表示在原数据上进行修改

    #1.4. 修改列名，将Churn_Yes改为flag 做为标签    False表示不流失，True表示流失
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)


#2. 定义函数，实现数据的可视化
def data_visualization():
    #1.1. 读取csv文件，获取到df对象
    data = pd.read_csv('./LogisticRegression/data/churn.csv')
    # data.info() #查看数据集信息

    #1.2. 由于csv文件中的Churn列和gender列是字符串类型，需要进行one-hot编码(热编码处理)
    data = pd.get_dummies(data,columns=['Churn','gender'])
    # data.info() #查看数据集信息

    #1.3. 删除one-hot编码处理后冗余的列
    data.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True) # axis=1表示删除列，inplace=True表示在原数据上进行修改

    #1.4. 修改列名，将Churn_Yes改为flag 做为标签    False表示不流失，True表示流失
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)

    # 数据可视化 绘制 计数柱状图
    # 参1：数据集 参2：x轴的列名(这里是 字段：月度会员) 参3：hue：根据什么字段进行分类 这里是是否流失
    sns.countplot(data = data, x = 'Contract_Month' , hue = 'flag')
    plt.show()

#3. 定义函数，实现逻辑回归模型训练
def logistic_regression():

    # 1. 读取csv文件，获取到df对象
    data = pd.read_csv('./LogisticRegression/data/churn.csv')
    # data.info() #查看数据集信息

    # 2.1 由于csv文件中的Churn列和gender列是字符串类型，需要进行one-hot编码(热编码处理)
    data = pd.get_dummies(data,columns=['Churn','gender'])
    # data.info() #查看数据集信息

    # 2.2 删除one-hot编码处理后冗余的列
    data.drop(['Churn_No', 'gender_Male'], axis=1, inplace=True) # axis=1表示删除列，inplace=True表示在原数据上进行修改

    # 2.3 修改列名，将Churn_Yes改为flag 做为标签    False表示不流失，True表示流失
    data.rename(columns={'Churn_Yes':'flag'}, inplace=True)

    # 2.4 提取特征列 和 标签列
    # x 的特征列：月度会员，是否有互联网服务，是否是电子支付
    x = data[['Contract_Month', 'internet_other', 'PaymentElectronic']]
    y = data['flag'] # False表示不流失，True表示流失

    # 2.5 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    # 3. 特征工程(特征提取、特征预处理：归一化、标准化...) 这里不做处理

    # 4. 模型训练
    # 4.1 创建逻辑回归模型
    estimator = LogisticRegression()
    # 4.2 模型训练
    estimator.fit(x_train, y_train)
    # 5 模型预测
    y_pred = estimator.predict(x_test)
    print('预测结果：', y_pred)

    # 6. 模型评估
    print('训练集准确率准确率：', estimator.score(x_test, y_test)) 
    print('测试集准确率准确率：', accuracy_score(y_test, y_pred))  

    print(f'精确率：{precision_score(y_test, y_pred)}')
    print(f'召回率：{recall_score(y_test, y_pred)}')
    print(f'F1值：{f1_score(y_test, y_pred)}')

    # macro avg: 宏平均，不考虑样本权重, 直接求平均，适用于数据均衡的情况
    # weighted avg: 加权平均，考虑样本权重, 适用于数据不均衡的情况
    print(f'分类评估报告：\n{classification_report(y_test, y_pred)}')


#4. 测试
if __name__ == '__main__':
    # data_preprocess()
    # data_visualization()
    logistic_regression()
```








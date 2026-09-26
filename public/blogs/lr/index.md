#  一、线性回归介绍

线性回归算法：属于有监督学习，有特征有标签。属于回归问题，标签连续

## 1.1 什么是线性回归？

线性回归（Linear Regressor）：利用回归方程，对1个或多个自变量（特征值）和因变量（目标值）之间的关系进行建模的一种方式。

属于有监督学习，有特征，有标签，且标签连续。

数学公式如下，其中T是线性代数中的转置
$$
h_{(w)} =w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + ...... + b = w^{T}x + b
$$
主要分为：一元线性 和 多元线性

## 1.2 一元线性与多元线性

### 1.2.1 一元线性回归

一元线性回归： 目标值只和一个因变量有关，简单来说就是一次函数，这里w叫权重，b叫偏置
$$
y = wx + b
$$
**1个**特征列 + 1个标签列

### 1.2.2 多元线性回归

多元线性回归：目标值与多个因变量有关
$$
h_{(w)} =w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + ...... + b = w^{T}x + b
$$
**多个**特征列 + 1个标签列

# 二、线性回归问题的求解

## 2.1 线性回归API

### 2.1.1 API调用过程

![image-20260925102505953](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925102505953.png)

### 2.1.2 入门代码

数据集如下：

![image-20260925102358805](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925102358805.png)

代码如下：

```python
#导包
from sklearn.linear_model import LinearRegression
#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

#1. 准备数据
x_train = [[160], [166], [172], [174], [180]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [[176]]

# 2. 数据的预处理 这里不需要
#3. 特征工程 包括特征提取、特征预处理 这里不需要

#4. 模型训练
model = LinearRegression() # 创建模型对象
model.fit(x_train, y_train) # 模型训练
#  可以查看一下权重和偏置
print(f'权重：{model.coef_}')
print(f'偏置：{model.intercept_}')

#5. 模型预测
y_predict = model.predict(x_test)
print(y_predict)

#6. 模型评估
```

# 三、 损失函数

**损失函数**(Loss Function，也叫成本函数、代价函数、目标函数)：是用于描述每个样本点 和 其预测值 之间的关系，用于在训练时指导模型调整参数。数值上 是**各个样本的误差和**，损失函数值**越小**，模型**越准**。

误差：预测值y - 真实值y

## 3.1 损失函数的种类

 损失函数有两类：均方误差MSE(Mean-Square Error)、平均绝对误差MAE(Mean Absolute Error)

**均方误差MSE**：计算公式如下，其中$y_{pred}$是预测值，$y_{true}$是样本值，n为样本数量。
$$
MSE= \frac{1}{n}\sum_{i = 1}^{n}(y_{pred} - y_{true}) ^{2}
$$
**平均绝对误差MAE**：计算公式如下：其中![y_{pred}](https://latex.csdn.net/eq?y_%7Bpred%7D)是预测值，![y_{true}](https://latex.csdn.net/eq?y_%7Btrue%7D)是样本值，n为样本数量。
$$
MAE = \frac{1}{n}\sum_{i = 1}^{n} |y_{pred} - y_{true} |
$$


## 3.2 损失函数求解方法

原理层面，了解即可

### 3.2.1正规方程法

多元线性回归：

![点击并拖拽以移动](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925103106624.png)

要求J(w)的最小值，J(w)对w求导，得到上图结果(1)，最终推出结果(8)

带入实例演示：X就是特征列矩阵

![点击并拖拽以移动](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925103200967.png)

存在的问题：

1、如果运算量过大，可能造成内存溢出。

2、假设矩阵没有逆，可能无解。

### 3.2.2 梯度下降法（主流）

梯度(grad)：一元函数中，就是某一点的导数，有方向为：函数值上升最快的方向

​          			多元函数中，就是某一点的偏导数，梯度是所有偏导数组成的向量

梯度下降算法：沿着梯度下降的方向求解极小值

![点击并拖拽以移动](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925103612893.png)

梯度下降公式：
$$
\theta _{i + 1} = \theta _{i} - \alpha \frac{\partial J(\theta )}{\partial \theta _{i}}
$$
循环迭代求当前点的梯度，更新当前权重参数

$\theta _{i + 1}$是下个点，$\theta _{i}$是上个点，$J(\theta)$是损失函数，$\frac{\partial J(\theta )}{\partial \theta _{i}}$对损失函数求偏导

$\alpha$：学习率(步长)，大小适度，在机器学习中常为0.001~0.01

减法是因为梯度是上升最快的方向，加上负号，变成下降最快方向。

重复直到收敛：两次迭代的差小于阈值 或 达到迭代次数

多变量实例：

![点击并拖拽以移动](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925103947023.png)

##### 3.2.2.1 梯度下降法案例

设 姓名x1、每月工资x2、存款余额x3、房产面积x4，授信额度 是 标签y

![点击并拖拽以移动](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925104213330.png)

这里的权重w换成了$\theta$

![image-20260925104330953](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925104330953.png)

分母加2是方便计算，简化编程。系数不影响极值点的位置，最后要让导数＝0

![image-20260925104558262](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925104558262.png)

![image-20260925104613533](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925104613533.png)

上面图中划掉，应该是损失函数的偏导(梯度)

下面进行带值，假设每一列的权重相同

![image-20260925104806889](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925104806889.png)

![image-20260925104819910](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925104819910.png)

一共8个人的数据，每个数据有4个分量(分别计算4个梯度)。

## 3.3 梯度下降法 分类

分为4类：

1、全梯度下降算法FGD（Full Gradient Descent）：每次迭代，使用全部样本的梯度值

缺点：使用全部数据集，训练速度较慢

2、随机梯度下降算法SGD：每次迭代，随机使用一个样本的梯度值

优点：简单、高效

缺点：不稳定，遇上噪声容易陷入局部最优解。

3、小批量梯度下降算法mini-batch：每次迭代，随机选择小批量的样本梯度值。

介于FGD和SGD之间，目前使用最多。

4、随机平均梯度下降算法SAG：每次迭代，随机选择一个样本的梯度值和以往样本的梯度值的均值。

缺点：训练初期表现不佳，优化速度较慢，因为常常将初始梯度设置为0



梯度下降与正规方程的对比

![image-20260925104935935](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925104935935.png)

# 四、回归模型评估方法

目的：衡量 预测值 和 真实值 之间的差距

## 4.1 均方误差**MSE**

**均方误差MSE**：计算公式如下，其中$y_{pred}$是预测值，$y_{true}$是样本值，n为样本数量。
$$
MSE= \frac{1}{n}\sum_{i = 1}^{n}(y_{pred} - y_{true}) ^{2}
$$
MSE越小，模型预测越准确

## 4.2 平均绝对误差**MAE**

**平均绝对误差MAE**：计算公式如下：其中$y_{pred}$是预测值，$y_{true}$是样本值，n为样本数量。
$$
MAE = \frac{1}{n}\sum_{i = 1}^{n} |y_{pred} - y_{true} |
$$
MAE越小，模型预测越准确

## 4.3 均方根误差

均方根误差**RMSE**：计算公式如下：其中$y_{pred}$是预测值，$y_{true}$是样本值，n为样本数量。
$$
RMSE= \sqrt{\frac{1}{n}\sum_{i = 1}^{n}(y_{pred} - y_{true}) ^{2}}
$$
RMSE越小，模型预测越准确。 RMSE会对异常点更加敏感

一般使用MAE和RMSE两个指标，一块使用并且评估 

# 五、线性回归API和案例

## 5.1 正规方程API

![image-20260925105838164](C:/Users/86176/AppData/Roaming/Typora/typora-user-images/image-20260925105838164.png)

## 5.2 梯度下降API

![image-20260925105935457](C:/Users/86176/AppData/Roaming/Typora/typora-user-images/image-20260925105935457.png)

## 5.3 案例：波士顿房价预测

数据说明：

![image-20260925105954066](C:/Users/86176/AppData/Roaming/Typora/typora-user-images/image-20260925105954066.png)

数据大小不一致，可能会对结果影响较大，需要标准化处理

### 5.3.1 代码实现

先使用正规方程法

#### 5.3.1.1 导入各种库

代码如下：

```python
from sklearn.preprocessing import StandardScaler # 导入标准化数据模块
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.linear_model import LinearRegression # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error # 均方误差评估

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字
```

#### 5.3.1.2 加载数据集

代码如下：

```python
#1、导入 波士顿房价 数据集
import pandas as pd
import numpy as np
import ssl

# 创建不验证证书的 SSL 上下文
ssl._create_default_https_context = ssl._create_unverified_context

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 打印部分数据
print(f'特征:{data.shape}') # (506, 13)506行，13列
print(f'标签:{target.shape}') # (506,) 506行
print(f'特征数据集:{data[:5]}') # 打印前5行特征数据
print(f'标签数据集:{target[:5]}') # 打印前5行标签数据
```

#### 5.3.1.3 数据的预处理

代码如下：

```python
#2、数据预处理 --切分 训练集 和 测试集
# 参1：特征数据 参2：标签数据 参3：测试集所占比例 参4：随机种子
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=23)
```

#### 5.3.1.4 特征工程

代码如下：

```python
#3、特征工程(特征提取、特征预处理...)
#3.1 创建标准化对象
transfer = StandardScaler()
#3.2 训练集标准化
x_train = transfer.fit_transform(x_train)
#3.3 测试集标准化
x_test = transfer.transform(x_test)
```

#### 5.3.1.5 模型训练

代码如下：

```python
#4、模型训练
#4.1 创建 线性回归 正规方程 的模型对象
estimator = LinearRegression(fit_intercept=True) #fit_intercept: 是否需要截距,默认为True
#4.2 训练模型
estimator.fit(x_train, y_train)
#4.3打印模型计算出的w(权重)和b(偏置)
print(f'正规方程法模型参数w:{estimator.coef_}') # 模型参数w
print(f'正规方程法模型参数b:{estimator.intercept_}') # 模型参数b
```

#### 5.3.1.6 模型预测

代码如下：

```python
#5、模型预测
y_pre = estimator.predict(x_test)
print(f'正规方程法模型预测值:{y_pre}') # 模型预测值
```

#### 5.3.1.7 模型评估

代码如下：

```python
#6、模型评估
# 参1：测试集标签 参2：预测结果
print(f'均方误差:{mean_squared_error(y_test, y_pre)}') # MSE:均方误差
print(f'均方根误差:{root_mean_squared_error(y_test, y_pre)}') # RMSE:均方根误差
print(f'平均绝对误差:{mean_absolute_error(y_test, y_pre)}') # MAE:平均绝对误差
```

正规方程法，完整代码如下：

```python
from sklearn.preprocessing import StandardScaler # 导入标准化数据模块
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.linear_model import LinearRegression # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error # 均方误差评估
from sklearn.metrics import mean_absolute_error # 平均绝对误差评估

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

#1、导入 波士顿房价 数据集
import pandas as pd
import numpy as np

import ssl
# 创建不验证证书的 SSL 上下文
ssl._create_default_https_context = ssl._create_unverified_context

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# 打印部分数据
print(f'特征:{data.shape}') # (506, 13)506行，13列
print(f'标签:{target.shape}') # (506,) 506行
print(f'特征数据集:{data[:5]}') # 打印前5行特征数据
print(f'标签数据集:{target[:5]}') # 打印前5行标签数据

#2、数据预处理 --切分 训练集 和 测试集
# 参1：特征数据 参2：标签数据 参3：测试集所占比例 参4：随机种子
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=23)

#3、特征工程(特征提取、特征预处理...)
#3.1 创建标准化对象
transfer = StandardScaler()
#3.2 训练集标准化
x_train = transfer.fit_transform(x_train)
#3.3 测试集标准化
x_test = transfer.transform(x_test)

#4、模型训练
#4.1 创建 线性回归 正规方程 的模型对象
estimator = LinearRegression(fit_intercept=True) #fit_intercept: 是否需要截距,默认为True
#4.2 训练模型
estimator.fit(x_train, y_train)
#4.3打印模型计算出的w(权重)和b(偏置)
print(f'正规方程法模型参数w:{estimator.coef_}') # 模型参数w
print(f'正规方程法模型参数b:{estimator.intercept_}') # 模型参数b

#5、模型预测
y_pre = estimator.predict(x_test)
print(f'正规方程法模型预测值:{y_pre}') # 模型预测值

#6、模型评估
# 参1：测试集标签 参2：预测结果
print(f'均方误差:{mean_squared_error(y_test, y_pre)}') # MSE:均方误差
print(f'均方根误差:{root_mean_squared_error(y_test, y_pre)}') # RMSE:均方根误差
print(f'平均绝对误差:{mean_absolute_error(y_test, y_pre)}') # MAE:平均绝对误差
```

对于梯度下降法：

只需要修改一下模型创建部分，代码如下：

```python
#4、模型训练
#4.1 创建 线性回归 梯度下降 的模型对象
# 参1：是否计算偏置 参2：学习率模式 constant常量(不会发生改变) 参3：学习率 参4：最大迭代次数
estimator = SGDRegressor(fit_intercept=True, learning_rate='constant', eta0=0.01, max_iter=1000)
#4.2 训练模型
estimator.fit(x_train, y_train)
#4.3打印模型计算出的w(权重)和b(偏置)
print(f'模型参数w:{estimator.coef_}') # 模型参数w
print(f'模型参数b:{estimator.intercept_}') # 模型参数b
```

完整代码如下：

```python
from sklearn.preprocessing import StandardScaler # 导入标准化数据模块
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.linear_model import LinearRegression # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error # 均方误差评估
from sklearn.metrics import mean_absolute_error # 平均绝对误差评估

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

#1、导入 波士顿房价 数据集
import pandas as pd
import numpy as np

import ssl
# 创建不验证证书的 SSL 上下文
ssl._create_default_https_context = ssl._create_unverified_context

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# 打印部分数据
print(f'特征:{data.shape}') # (506, 13)506行，13列
print(f'标签:{target.shape}') # (506,) 506行
print(f'特征数据集:{data[:5]}') # 打印前5行特征数据
print(f'标签数据集:{target[:5]}') # 打印前5行标签数据

#2、数据预处理 --切分 训练集 和 测试集
# 参1：特征数据 参2：标签数据 参3：测试集所占比例 参4：随机种子
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=23)

#3、特征工程(特征提取、特征预处理...)
#3.1 创建标准化对象
transfer = StandardScaler()
#3.2 训练集标准化
x_train = transfer.fit_transform(x_train)
#3.3 测试集标准化
x_test = transfer.transform(x_test)

#4、模型训练
#4.1 创建 线性回归 梯度下降 的模型对象
# 参1：是否计算偏置 参2：学习率模式 constant常量(不会发生改变) 参3：学习率 参4：最大迭代次数
estimator = SGDRegressor(fit_intercept=True, learning_rate='constant', eta0=0.01, max_iter=1000)
#4.2 训练模型
estimator.fit(x_train, y_train)
#4.3打印模型计算出的w(权重)和b(偏置)
print(f'模型参数w:{estimator.coef_}') # 模型参数w
print(f'模型参数b:{estimator.intercept_}') # 模型参数b

#5、模型预测
y_pre = estimator.predict(x_test)
print(f'模型预测值:{y_pre}') # 模型预测值

#6、模型评估
# 参1：测试集标签 参2：预测结果
print(f'均方误差:{mean_squared_error(y_test, y_pre)}') # MSE:均方误差
print(f'均方根误差:{root_mean_squared_error(y_test, y_pre)}') # RMSE:均方根误差
print(f'平均绝对误差:{mean_absolute_error(y_test, y_pre)}') # MAE:平均绝对误差
```

# 六、过拟合和欠拟合

 过拟合：模型在训练集表现**好**、测试集表现**差**

过拟合原因：模型过于复杂、数据少或训练过度。

解决办法：重新清洗数据、增大数据训练量、**正则化**、减少特征维度

欠拟合：模型在训练集表现**差**、测试集表现**差**

欠拟合原因：模型过于简单、特征不足。

解决办法：添加特征列、组合 泛化 相关性、添加多项式特征项

## 6.1 绘图展示

### 6.1.1 欠拟合

代码如下：

```python
def under_fitting():
    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 使用正规方程训练模型
    estimator = LinearRegression()
    #4.2 模型训练
    estimator.fit(X, y)

    #5. 模型预测
    y_pre = estimator.predict(X)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    plt.plot(x, y_pre, color='r') # 折线图 绘制预测值
    plt.show()
```

绘制图片如下：

![image-20260925120455498](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925120455498.png)

### 6.1.2 正好拟合

代码如下：

```python
#2. 定义函数，模拟：拟合
# 只需要改动数据预处理
def fitting():
    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，会出现欠拟合现象。这里增加1个特征列 增加模型复杂度
    X2 = np.hstack([X, X ** 2]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 使用正规方程训练模型
    estimator = LinearRegression()
    #4.2 模型训练
    estimator.fit(X2, y)

    #5. 模型预测
    y_pre = estimator.predict(X2)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()
```

绘制图片如下：

![image-20260925120519432](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925120519432.png)

### 6.1.3 过拟合

代码如下：

```python
#3. 定义函数，模拟：过拟合
def over_fitting():
    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，为了模拟过拟合现象，新增9列增加模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 使用正规方程训练模型
    estimator = LinearRegression()
    #4.2 模型训练
    estimator.fit(X3, y)

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()
```

绘制图片如下：

![image-20260925120603232](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925120603232.png)

## 6.2 正则化

用于解决过拟合问题，分为L1正则化、L2正则化。开发中一般使用L2正则

### 6.2.1 L1正则化

L1正则化：在损失函数中添加L1正则化项，公式如下：
$$
J(w) = MSE(w) + \alpha \sum_{i = 1}^{n}\left | w_{i} \right |
$$
补充一下 MSE均方误差 的公式
$$
MSE= \frac{1}{n}\sum_{i = 1}^{n}(y_{pred} - y_{true}) ^{2}
$$
![\alpha](https://latex.csdn.net/eq?%5Calpha)叫做**惩罚系数**，值越大，权重调整幅度越大

L1正则化会使权重趋向于0(可以=0)，使得某些特征失效，达到**特征筛选**的目的

#### 6.2.1.1 代码实现

代码如下：

```python
def l1_regularization():
     # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，为了模拟过拟合现象，新增9列增加 模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 创建L1正则化对象
    estimator = Lasso(alpha=0.1) # alpha:正则化系数(惩罚系数) 默认1
    #4.2 模型训练
    estimator.fit(X3, y)

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()
```

结果图如下：

![image-20260925120724622](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925120724622.png)

### 6.2.2 L2正则化

L2正则化：在损失函数中添加L2正则化项，公式如下：
$$
J(w) = MSE(w) + \alpha \sum_{i = 1}^{n}w_{i}^{2}
$$
![\alpha](https://latex.csdn.net/eq?%5Calpha)叫做**惩罚系数**，值越大，权重调整幅度越大

L2正则化会使权重趋向于0(一般≠0)

使用L2正则化的线性回归模型是**岭回归**

#### 6.2.2.1 代码实现

代码如下：

```python
def l2_regularization():

    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，为了模拟过拟合现象，新增9列增加 模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 创建L2正则化对象
    estimator = Ridge(alpha=10) # alpha:正则化系数(惩罚系数) 默认1
    #4.2 模型训练
    estimator.fit(X3, y)

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()
```

结果如图：

![image-20260925120816810](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925120816810.png)

过拟合和欠拟合的完整代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.linear_model import LinearRegression # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, root_mean_squared_error # 均方误差评估
from sklearn.metrics import mean_absolute_error # 平均绝对误差评估
from sklearn.linear_model import Lasso, Ridge # L1正则化回归模型 L2正则化回归模型

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

#1. 定义函数，模拟：欠拟合
def under_fitting():
    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 使用正规方程训练模型
    estimator = LinearRegression()
    #4.2 模型训练
    estimator.fit(X, y)

    #5. 模型预测
    y_pre = estimator.predict(X)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    plt.plot(x, y_pre, color='r') # 折线图 绘制预测值
    plt.show()

#2. 定义函数，模拟：拟合
# 只需要改动数据预处理
def fitting():
    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，会出现欠拟合现象。这里增加1个特征列 增加模型复杂度
    X2 = np.hstack([X, X ** 2]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 使用正规方程训练模型
    estimator = LinearRegression()
    #4.2 模型训练
    estimator.fit(X2, y)

    #5. 模型预测
    y_pre = estimator.predict(X2)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()


#3. 定义函数，模拟：过拟合
def over_fitting():
    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，为了模拟过拟合现象，新增9列增加模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 使用正规方程训练模型
    estimator = LinearRegression()
    #4.2 模型训练
    estimator.fit(X3, y)

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()

#4. 定义函数，模拟：l1正则化
def l1_regularization():
    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，为了模拟过拟合现象，新增9列增加 模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 创建L1正则化对象
    estimator = Lasso(alpha=0.1) # alpha:正则化系数(惩罚系数) 默认1
    #4.2 模型训练
    estimator.fit(X3, y)

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()

# 5. 定义函数，模拟：l2正则化
def l2_regularization():

    # 1. 准备数据
    # 1.1 指定随机种子，保证每次生成结果一致
    np.random.seed(23)
    # 1.2 随机生成x轴 100个数据，模拟：特征
    x = np.random.uniform(-3, 3, 100) # 从-3到3中随机生成100个数据
    # 1.3 基于x轴的值，随机生成y轴 100个数据，模拟：标签
    # y = kx + b = 0.5 * x ** 2 + x + 2 + 噪声 这里k和b的值是随便取的
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, 100) # 噪声：均值为0，标准差为1 生成100个
    # 1.4 查看生成的数据
    print(f'特征(x):{x}') 
    print(f'标签(y):{y}')

    #2. 数据预处理，把x轴(特征)转化成多行1列的形式
    X = x.reshape(-1, 1)

    #2.1 由于模型只有一列过于简单，为了模拟过拟合现象，新增9列增加 模型复杂度
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10]) # 函数作用:水平拼接，行数不变，列数增加
    print(f'特征(X):{X}')

    #3. 特征工程，这里不做，直接使用

    #4. 模型训练
    #4.1 创建L2正则化对象
    estimator = Ridge(alpha=10) # alpha:正则化系数(惩罚系数) 默认1
    #4.2 模型训练
    estimator.fit(X3, y)

    #5. 模型预测
    y_pre = estimator.predict(X3)

    #6. 模型评估
    print(f'均方误差:{mean_squared_error(y, y_pre)}') # MSE:均方误差
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}') # RMSE:均方根误差
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}') # MAE:平均绝对误差

    #7. 绘图
    plt.scatter(x, y) # 散点图 绘制真实值
    # np.sort(x) # 对x轴进行排序 np.argsort(x) 对x轴进行排序,返回排序后的索引
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r') # 折线图 绘制预测值
    plt.show()

#6. 测试
if __name__ == '__main__':
    # under_fitting() # 欠拟合
    # fitting() # 拟合
    # over_fitting() # 过拟合
    # l1_regularization() # l1正则化
    l2_regularization() # l2正则化
```






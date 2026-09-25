参考课程：黑马https://www.bilibili.com/video/BV1Fzszz4Ek7?spm_id_from=333.788.videopod.episodes&vd_source=24c1e92bdfe1c6a0f1b228cda0583ac9

# 一、KNN算法介绍

## 1.1 算法思想

KNN算法，即K-Nearest Neighbors，也叫K近邻算法

KNN分为2种，分类问题（标签不连续）和回归问题（标签连续），分类是投票，回归是均值

**KNN 算法思想：**通过**度量样本间的距离**，在特征空间中选择与待预测样本==最相似的 K 个样本==，依据这 K 个样本的信息进行投票或平均，以此完成分类或回归预测。

**样本相似性：**度量样本间的距离，距离越近相似度越高。

**度量距离的方法：**欧氏距离法 等。



## 1.2 K值大小

**K值过小：**容易受到异常点影响。而且K值过小会使模型变得相对复杂，发生==过拟合==现象

**K值过大：**容易受到样本均值影响。而且K值过大会让模型变得相对简单，发生==欠拟合==现象。



## 1.3 分类与回归

### 1.3.1 分类流程

1、计算 目标样本 到每一个训练样本的距离

2、将训练样本按距离 ==从小到大== 升序排列

3、取出距离最近的K个训练样本

4、进行 ==多数表决==，统计K个样本中哪个类别样本出现的最多

5、将 目标样本 归到 ==出现次数最多==的样本中。

**总结：**分类就是投票，根据距离找最近K个，然后将K个分类统计，如果相同，取最小的



### 1.3.2 回归流程

1、计算 目标样本 到每一个训练样本的距离

2、将训练样本按距离 ==从小到大== 升序排列

3、取出距离最近的K个训练样本

4、计算这K个样本对应标签的==平均值==

5、将平均值作为 目标样本 的预测值

**总结：**回归是均值，根据距离找最近K个，将K个的标签相加取平均值。



# 二、KNN算法实现

## 2.1 分类问题

数据样例如下：

| **x_train（特征）** |   **y_train(标签)**   |
| :-----------------: | :-------------------: |
|          0          |           0           |
|          1          |           0           |
|          2          |           1           |
|          3          |           1           |
|     **x_test**      | **y_predict(y_test)** |
|          5          |          ？           |

Python代码样例如下：

```python
# 1、导包，Classifier：分类 
from sklearn.neighbors import KNeighborsClassifier

#VS Code 终端默认编码不是 UTF-8 
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 2、准备数据集(训练集 和 测试集)
# 训练集
x_train = [[0], [1], [2], [3]]  # 训练集特征数据,因为特征可以有多个，所以是一个二维数组

# 训练集标签
y_train = [0, 0, 1, 1]  # 训练集标签数据，标签是离散的，所以是一个一维数组

# 测试集
x_test = [[5]]  # 测试集特征数据

# 3、创建KNN 分类模型 对象, n_neighbors：指定k值
model = KNeighborsClassifier(n_neighbors=2)  # 创建KNN模型对象

# 4、模型训练
#fit：拟合，告诉模型通过x_train可以得到y_train
model.fit(x_train, y_train)  # 模型训练

# 5、模型预测
y_predict = model.predict(x_test)  # 模型预测

# 6、打印预测结果
print("预测结果：", y_predict)  # 打印预测结果
```

结果应该是：

![image-20260925091701518](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925091701518.png)



## 2.2 回归问题

数据如下：

![image-20260925084406108](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925084406108.png)

代码如下：

```python
# 1、导包，Regressor:回归器
from sklearn.neighbors import KNeighborsRegressor

#VS Code 终端默认编码不是 UTF-8 
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 2、准备数据集(训练集 和 测试集)
# 训练集
x_train = [
    [0, 0, 1],
    [1, 1, 0],
    [3, 10, 10],
    [4, 11, 12]
]# 训练集特征数据
# 差值  (3, 11, 9)      (2, 10, 10)     (3, 10, 10)     (4, 11, 12)
# 平方和    211             204             1               5
# 开根号    14.53           14.28           1               2.24
# 结果应该是0.3 

# 训练集标签
y_train = [0.1, 0.2, 0.3, 0.4]# 训练集标签数据，标签是离散的，所以是一个一维数组

# 测试集
x_test = [[3, 11, 10]]  # 测试集特征数据

# 3、创建KNN 回归模型 对象, n_neighbors：指定k值
model = KNeighborsRegressor(n_neighbors=3)  # 创建KNN模型对象

# 4、模型训练
#fit：拟合，告诉模型通过x_train可以得到y_train
model.fit(x_train, y_train)  # 模型训练

# 5、模型预测
y_predict = model.predict(x_test)  # 模型预测

# 6、打印预测结果
print("预测结果：", y_predict)  # 打印预测结果
```

结果如下：

![image-20260925091459797](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925091459797.png)

# 三、常用度量方法

常用度量方法有：欧氏距离、曼哈顿距离、切比雪夫距离、闵可夫斯基距离等。

## 3.1 欧式距离

**欧氏距离：**两点间在空间中的直线距离：
$$
d(x, y) = \sqrt{\sum_{i = 1}^{n} (x_{i} - y_{i})^{2}}
$$

## 3.2 曼哈顿距离

**曼哈顿距离：**也叫**城市街区距离**，是对应维度差值的绝对值之和：
$$
d(x, y) =\sum_{i = 1}^{n} |x_{i} - y_{i}|
$$

## 3.3 切比雪夫距离

**切比雪夫距离：**各维度差值的最大值。即在二维平面中，从起点开始，可以延8个方向移动1个单位，记为1步，从起点到终点的最短步数就是切比雪夫距离。

![image-20260925091835780](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925091835780.png)

假设一共n个维度，i从1到n，公式如下：
$$
d(x, y) = max_{i}(|x_{i} - y_{i}|)
$$

## 3.4 闵可夫斯基距离

**闵可夫斯基距离：**又称闵氏距离。不是新的距离方式，而是对多个距离公式的概括。距离公式为：
$$
d(x, y) = \sqrt[p]{\sum_{n}^{i = 1} |x_{i} - y_{i}|^{p}}
$$
p是一个变参数

p = 1时，就是曼哈顿距离

p = 2时，就是欧式距离

p->∞时，就是切比雪夫距离

# 四、特征预处理

为什么需要归一化和标准化？

特征的单位 或 大小 相差较大，或者 某特征的方差相比其他特征要大好几个数量级，容易影响（支配）结果。

如下图，体重的数量级明显大于身高和视力的。

![image-20260925092030062](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925092030062.png)

## 4.1 归一化

### 4.1.1 归一化计算

**归一化：**通过对原始数据进行变换，将数据映射到 [ml, mr] 区间中（默认[0, 1]）

计算公式为：（当前值 - 该列最小值） / （该列最大值 - 该列最小值）：
$$
x^{'} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$
这个算完后，区间为[0, 1]

如果想映射到指定区间[ml, mr]，再进行计算：
$$
x^{''} = x^{'} * (mr - ml) + ml
$$
**弊端：**容易受到 最大值 和 最小值 的影响，所以一般用于处理 小数据集。

### 4.1.2代码样例

样例数据：

| 90   | 2    | 10   | 40   |
| ---- | ---- | ---- | ---- |
| 60   | 4    | 15   | 45   |
| 75   | 3    | 13   | 46   |

代码实现：

```python
# 1、导入 预处理包, StandardScaler：标准化，MinMaxScaler：归一化
from sklearn.preprocessing import  MinMaxScaler, StandardScaler 

#VS Code 终端默认编码不是 UTF-8 
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 2、准备数据集(归一化之前原数据)
x_train = [
    [90, 2, 10, 40],
    [60, 4, 15, 45],
    [75, 3, 13, 46]
]

# 3、创建归一化对象
# feature_range=(ml, mr) 表示将数据归一化到 [ml, mr] 之间
# 括号内参数不写默认[0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# 4、对原数据集进行归一化操作
x_train_new = scaler.fit_transform(x_train)

# 5、打印归一化后的数据
print("归一化的数据集为: \n")
print(x_train_new)
```

结果为：

![image-20260925092135258](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925092135258.png)

## 4.2 标准化

### 4.2.1 标准化计算

数据标准化：通过对原始数据进行标准化，转换为均值为0、标准差为1 的标准正态分布的数据。

![\mu](https://latex.csdn.net/eq?%5Cmu)为这一特征均值（该列平均值），![\sigma](https://latex.csdn.net/eq?%5Csigma)为特征的标准差（方差开平方，该列的的标准差），计算公式为：
$$
x' = \frac{x - \mu }{\sigma }
$$
应用场景：适用于 大数据集 的处理

### 4.2.2 代码样例

样例数据：

| 90   | 2    | 10   | 40   |
| ---- | ---- | ---- | ---- |
| 60   | 4    | 15   | 45   |
| 75   | 3    | 13   | 46   |

样例代码：

```python
# 1、导入 预处理包, StandardScaler：标准化，MinMaxScaler：归一化
from sklearn.preprocessing import  MinMaxScaler, StandardScaler 

#VS Code 终端默认编码不是 UTF-8 
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 2、准备数据集(标准化之前原数据)
x_train = [
    [90, 2, 10, 40],
    [60, 4, 15, 45],
    [75, 3, 13, 46]
]

# 3、创建标准化对象
scaler = StandardScaler()

# 4、对原数据集进行标准化操作
x_train_new = scaler.fit_transform(x_train)

# 5、打印标准化后的数据
print("标准化的数据集为: \n")
print(x_train_new)
```

样例结果：

![image-20260925092235509](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925092235509.png)

# 五、超参数选择方法

交叉验证 解决模型的数据输入问题（数据集划分），得到更加可靠的模型

网格搜索解决超参数的组合

二者的组合形成一个模型参数调优的解决方案。

## 5.1 交叉验证

**交叉验证：**是一种数据集的分割方法，将训练集划分为n份，拿一份做为验证集，其余n-1份做为训练集。又称n折交叉验证。目的是为了得到更加准确可信的模型评分

若分为4份，即n = 4

第一次：将第1份作为验证集，其余作为训练集

第二次：将第2份作为验证集，其余作为训练集

第三次：将第3份作为验证集，其余作为训练集

第n次：将第n份作为验证集，其余作为训练集

使用训练集＋验证集多次评估模型，取平均值做为交叉验证的模型得分

如果n = 4时模型得分最好，使用全部训练集（验证集 + 训练集 共n份）对n = 4的模型再训练一遍，再使用测试集（1份）对n = 4的模型进行评估。

## 5.2 网格搜索

**网格搜索：**模型之中有很多超参数，能力存在很大差异，需要手动产生很多超参数组合，来训练模型。将每一组超参数都使用交叉验证评估，最后选出最优超参数组合。网格调参是模型调参的有力工具，是寻找最优超参数的工具。

例：比如K就是一种超参，如果我们要测试k = 1， 2， 3， 5， 7哪个是最优超参，使用4折交叉验证。则一共需要执行20次程序才能找出来。

**实现过程：**只需要将若干参数传递给网格搜索对象，它自动帮我们完成不同参数的组合、模型训练、模型评估。最终返回一组最优的超参数。

## 5.3 API实现

网格搜索+交叉验证，本质上指的就是GridSearchCV这个API

API介绍：

```python
sklearn.model_selection.GridSearchCV(estimator, param_grid = None, cv = None)
```

estimator：模型对象

param_grid：估计器参数

cv：指定几折交叉验证

fit：输入训练数据

score：准确率

结果分析：

bestscore__：在交叉验证中最好的结果

bestestimato：最好的参数模型

cvresults：每次交叉验证后的验证集准确率结果和训练集准确率结果

在案例中会进行实践

# 六、案例实现

## 6.1 利用KNN算法对鸢尾花分类

### 6.1.1 题目分析

在python的sklearn的库中，有鸢尾花的数据集，可以直接调用

鸢尾花有3种类型，通过花萼的长、宽 和 花瓣的长、宽来判断种类。

![image-20260925092423983](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925092423983.png)

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

下面是鸢尾花的数据集，包括花萼的长、宽 和 花瓣的长、宽，一共4种特征，以及种类 一种标签

![image-20260925092622490](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925092622490.png)

属于有标签、有特征、不连续

选择使用KNN算法的分类方式实现。

### 6.1.2 项目流程

1、获取数据集

2、对数据进行基本处理

3、对数据进行预处理

4、选择合适模型，进行模型训练

5、模型评估

6、模型评测

### 6.1.3 代码实现

在**最后会放完整代码**，中间放的是代码片段，目的是进行练习。

#### 6.1.3.1 导入各种库

导入各种库，代码如下：

```python
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import seaborn as sns # 高级绘图库，用来画统计图、散点图、热力图，方便观察数据分布。
import pandas as pd # 用来处理数据，方便数据清洗、数据筛选、数据转换等操作。
import matplotlib.pyplot as plt # 用来画图，方便观察数据分布、模型预测结果等。
from sklearn.model_selection import train_test_split, GridSearchCV # 划分训练集和测试集 GridSearchCV寻找最优超参与网格搜索
from sklearn.preprocessing import StandardScaler # 数据标准化
from sklearn.neighbors import KNeighborsClassifier #导入 KNN 分类模型
from sklearn.metrics import accuracy_score # 模型评估，计算模型预测的准确率

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
```

#### 6.1.3.2 加载数据集

加载鸢尾花数据集，原始数据是字典形式，分别打印键值对看一下数据都包含了哪些内容。

练习内容方便理解，实际开发中不需要，下一次给代码就删掉这些了。

代码如下：

```python
# 1、定义函数形式，加载鸢尾花数据集，查看数据集
def dm01_loadiris():

    # 1.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 1.2 查看数据集
    # print(f'数据集：{iris_data}') # 字典形式
    # print(f'数据集的类型：{type(iris_data)}') 

    # 1.3 查看数据集的键
    print(f'数据集的键：{iris_data.keys()}')

    # 1.4 查看数据键对应的值
    print(f'数据集的键对应的值：{iris_data.data[:5]}') # 有150条数据，每条数据4个特征，查看前5行数据
    print(f'具体的标签：{iris_data.target[:5]}') # 有150条数据，每条数据1个标签，查看前5条数据的标签
    print(f'标签对应的名称：{iris_data.target_names}') # 标签的名称 ['setosa' 'versicolor' 'virginica']
    print(f'特征对应的名称：{iris_data.feature_names}') # 特征的名称 ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # print(f'数据集的描述：{iris_data.DESCR}') # 数据集的描述
    print(f'数据集的框架：{iris_data.frame}') # 数据集的框架
    print(f'数据集的文件名：{iris_data.filename}') # 数据集的文件名
    print(f'数据集的模型(在哪个包下)：{iris_data.data_module}') # 模型(在哪个包下)


# 5、测试
if __name__ == '__main__':
    dm01_loadiris()
```

数据集的键：dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

其中，只有data、target 、target_names、feature_names我们要用，对应关系如下：

![image-20260925094335340](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925094335340.png)

#### 6.1.3.3 图像绘制练习

上面的dm01函数可以删除了。根据数据进行绘制图像，

代码如下：

```python
def dm02_drawIris():
    # 2.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 2.2 将数据集封装为 DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

    # 2.3 先给df对象新增一列 标签列
    iris_df['target'] = iris_data.target

    # 2.4 使用seaborn库绘制散点图
    # data:数据集, x轴:花萼长度, y轴:花萼宽度, hue：根据标签列分类, fit_reg：是否绘制回归线
    sns.lmplot(data = iris_df, x = 'sepal length (cm)', y = 'sepal width (cm)', hue = 'target', fit_reg = False)

    # 2.5 设置标题
    plt.title('鸢尾花数据集的散点图')
    plt.tight_layout() # 自动调整子图参数, 使之填充整个图像区域
    plt.show()

# 5、测试
if __name__ == '__main__':
    dm01_loadIris()
    dm02_drawIris()
```

图像结果如下：

![image-20260925094356418](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925094356418.png)

#### 6.1.3.4 数据集划分

代码如下：

```python
def dm03_splitData():
    # 3.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 3.2 数据的预处理：从150个特征和标签中，按照8：2的比例划分训练集和测试集
    # 参数分别为：特征，标签，测试集比例， 随机种子：相同的随机种子，每次划分的结果是一样的
    # 返回值：训练集特征，测试集特征，训练集标签，测试集标签
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

    # 3.3打印切割后的数据集 
    print(f'训练集特征：{x_train}, 个数:{len(x_train)}') # 120条 每条4列 特征
    print(f'训练集标签：{y_train}, 个数:{len(y_train)}') # 120条 每条1列 标签
    print(f'测试集特征：{x_test}, 个数:{len(x_test)}')  # 30条 每条4列 特征
    print(f'测试集标签：{y_test}, 个数:{len(y_test)}') # 30条 每条1列 标签

# 5、测试
if __name__ == '__main__':
    # dm01_loadIris()
    # dm02_drawIris()
    dm03_splitData()
```

运行结果，内容较多，只展示部分测试集：

![image-20260925094422848](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925094422848.png)

#### 6.1.3.5 模型训练、预测、评估

代码如下：

```python
# 4、定义函数，模型测试、训练、预测、评估
def dm04_iris_evaluate_test():
    
    # 4.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 4.2 数据的预处理：从150个特征和标签中，按照8：2的比例划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=23)

    # 4.3 特征提取,由于原数据只有4个特征列，不需要特征提取
    # 4.4 数据标准化, 由于原数据的4列特征差值不大，不需要标准化，但是加入预处理代码更加完善
    scale = StandardScaler()

    # fit 是计算均值和方差，transform 是进行标准化
    # fit_transform 是兼具fit和transform功能， 先算均值方差，再标准化。适用于 第一次 标准化的训练，一般用于处理训练集
    x_train = scale.fit_transform(x_train)

    # 把训练集的均值、方差用到测试集，而不是重新计算，所以不加fit
    x_test = scale.transform(x_test)

    # 4.5 模型训练, k值要看需求
    model = KNeighborsClassifier(n_neighbors=3) # n_neighbors=3 表示3个邻居
    # 模型训练动作
    model.fit(x_train, y_train) # x_train是训练集的特征，y_train是训练集的标签

    # 4.6 模型预测

    # 4.6.1 对测试集进行预测
    y_pred = model.predict(x_test) # x_test是测试集的特征
    print(f'预测结果：{y_pred}')

    # 4.6.2 对新数据进行预测
    # 自定义测试数据集my
    my_data = [[7.8, 2.1, 3.9, 1.6]] # 新数据，需要和训练集的特征列一致
    my_data = scale.transform(my_data) # 对新数据进行标准化
    my_pred = model.predict(my_data) # 对新数据进行预测
    print(f'新数据预测结果：{my_pred}')
    # 查看上述数据集，每种分类的预测概率
    my_pred_proba = model.predict_proba(my_data)
    print(f'新数据预测概率：{my_pred_proba}')

    # 4.7 模型评估
    #4.7.1 直接评分 基于 训练集的特征 和 训练集的标签
    print(f'准确率: {model.score(x_train, y_train)}')

    # 4.7.2 基于 测试集的标签 和 预测结果
    print(f'准确率: {accuracy_score(y_test, y_pred)}')


# 5、测试
if __name__ == '__main__':
    # dm01_loadIris()
    # dm02_drawIris()
    # dm03_splitData()
    dm04_iris_evaluate_test()
```

结果如图：

![image-20260925094447702](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925094447702.png)

#### 6.1.3.6 超参数选择

通过交叉验证、网格搜索，寻找最优的K值，得到最优K值后带入计算准确率

代码如下：

```python
# 5、寻找最优超参
def dm05_find_best_k():
    # 5.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 5.2 数据的预处理：从150个特征和标签中，按照8：2的比例划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=23)
    # 5.3 特征提取,由于原数据只有4个特征列，不需要特征提取
    # 5.4 数据标准化, 由于原数据的4列特征差值不大，不需要标准化，但是加入预处理代码更加完善
    scale = StandardScaler()

    # fit 是计算均值和方差，transform 是进行标准化
    x_train = scale.fit_transform(x_train)
    # 把训练集的均值、方差用到测试集，而不是重新计算，所以不加fit   
    x_test = scale.transform(x_test)

    # 5.5 模型训练 与 寻找最优超参
    model = KNeighborsClassifier()
    param_dict = {'n_neighbors': [i for i in range(1, 11)]} # 超参字典，n_neighbors取值范围1-10

    # 创建 GridSearchCV对象
    # 三个参数：要计算最优超参的模型、该模型超参可能出现的值、交叉验证的折数 每个超参组合都要进行4次交叉验证，一共4*10=40次
    # 返回值：处理后的 最优超参模型对象
    model = GridSearchCV(model, param_dict, cv=4)

    # 模型训练动作
    model.fit(x_train, y_train) # x_train是训练集的特征，y_train是训练集的标签

    # 打印最优超参组合
    print(f'最优评分：{model.best_score_}')
    print(f'最优超参组合：{model.best_params_}')
    print(f'最优模型：{model.best_estimator_}')
    # print(f'具体的交叉验证结果：{model.cv_results_}')

    """
    上面得到的结果：
    最优评分：0.9583333333333334
    最优超参组合：{'n_neighbors': 7}
    最优模型：KNeighborsClassifier(n_neighbors=7)
    """
    # 模型评估
    # 对最优超参进行验证
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # 计算准确率
    print(f'准确率：{accuracy_score(y_test, y_pred)}')


# 5、测试
if __name__ == '__main__':
    # dm01_loadIris()
    # dm02_drawIris()
    # dm03_splitData()
    # dm04_iris_evaluate_test()
    dm05_find_best_k()
```

代码结果如下：

![image-20260925094505387](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260925094505387.png)

#### 6.1.3.7 完整代码

代码如下：

```python
# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import seaborn as sns # 高级绘图库，用来画统计图、散点图、热力图，方便观察数据分布。
import pandas as pd # 用来处理数据，方便数据清洗、数据筛选、数据转换等操作。
import matplotlib.pyplot as plt # 用来画图，方便观察数据分布、模型预测结果等。
from sklearn.model_selection import train_test_split, GridSearchCV # 划分训练集和测试集 GridSearchCV寻找最优超参与网格搜索
from sklearn.preprocessing import StandardScaler # 数据标准化
from sklearn.neighbors import KNeighborsClassifier #导入 KNN 分类模型
from sklearn.metrics import accuracy_score # 模型评估，计算模型预测的准确率

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1、定义函数形式，加载鸢尾花数据集，查看数据集
def dm01_loadIris():
    # 1.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 1.2 查看数据集
    print(f'数据集：{iris_data}') # 字典形式
    print(f'数据集的类型：{type(iris_data)}') 

    # 1.3 查看数据集的键
    print(f'数据集的键：{iris_data.keys()}')

    # 1.4 查看数据键对应的值
    print(f'数据集的键对应的值：{iris_data.data[:5]}') # 有150条数据，每条数据4个特征，查看前5行数据
    print(f'具体的标签：{iris_data.target[:5]}') # 有150条数据，每条数据1个标签，查看前5条数据的标签
    print(f'标签对应的名称：{iris_data.target_names}') # 标签的名称 ['setosa' 'versicolor' 'virginica']
    print(f'特征对应的名称：{iris_data.feature_names}') # 特征的名称 ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(f'数据集的描述：{iris_data.DESCR}') # 数据集的描述
    print(f'数据集的框架：{iris_data.frame}') # 数据集的框架
    print(f'数据集的文件名：{iris_data.filename}') # 数据集的文件名
    print(f'数据集的模型(在哪个包下)：{iris_data.data_module}') # 模型(在哪个包下)
    
# 2、定义函数，绘制数据集的散点图
def dm02_drawIris():
    # 2.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 2.2 将数据集封装为 DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

    # 2.3 先给df对象新增一列 标签列
    iris_df['target'] = iris_data.target

    # 2.4 使用seaborn库绘制散点图
    # data:数据集, x轴:花萼长度, y轴:花萼宽度, hue：根据标签列分类, fit_reg：是否绘制回归线
    sns.lmplot(data = iris_df, x = 'sepal length (cm)', y = 'sepal width (cm)', hue = 'target', fit_reg = False)

    # 2.5 设置标题
    plt.title('鸢尾花数据集的散点图')
    plt.tight_layout() # 自动调整子图参数, 使之填充整个图像区域
    plt.show()

# 3、定义函数，划分训练集和测试集
def dm03_splitData():
    # 3.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 3.2 数据的预处理：从150个特征和标签中，按照8：2的比例划分训练集和测试集
    # 参数分别为：特征，标签，测试集比例， 随机种子：相同的随机种子，每次划分的结果是一样的
    # 返回值：训练集特征，测试集特征，训练集标签，测试集标签
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

    # 3.3打印切割后的数据集 
    print(f'训练集特征：{x_train}, 个数:{len(x_train)}') # 120条 每条4列 特征
    print(f'训练集标签：{y_train}, 个数:{len(y_train)}') # 120条 每条1列 标签
    print(f'测试集特征：{x_test}, 个数:{len(x_test)}')  # 30条 每条4列 特征
    print(f'测试集标签：{y_test}, 个数:{len(y_test)}') # 30条 每条1列 标签

# 4、定义函数，模型测试、训练、预测、评估
def dm04_iris_evaluate_test():
    
    # 4.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 4.2 数据的预处理：从150个特征和标签中，按照8：2的比例划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=23)

    # 4.3 特征提取,由于原数据只有4个特征列，不需要特征提取
    # 4.4 数据标准化, 由于原数据的4列特征差值不大，不需要标准化，但是加入预处理代码更加完善
    scale = StandardScaler()

    # fit 是计算均值和方差，transform 是进行标准化
    # fit_transform 是兼具fit和transform功能， 先算均值方差，再标准化。适用于 第一次 标准化的训练，一般用于处理训练集
    x_train = scale.fit_transform(x_train)

    # 把训练集的均值、方差用到测试集，而不是重新计算，所以不加fit
    x_test = scale.transform(x_test)

    # 4.5 模型训练, k值要看需求
    model = KNeighborsClassifier(n_neighbors=3) # n_neighbors=3 表示3个邻居
    # 模型训练动作
    model.fit(x_train, y_train) # x_train是训练集的特征，y_train是训练集的标签

    # 4.6 模型预测

    # 4.6.1 对测试集进行预测
    y_pred = model.predict(x_test) # x_test是测试集的特征
    print(f'预测结果：{y_pred}')

    # 4.6.2 对新数据进行预测
    # 自定义测试数据集my
    my_data = [[7.8, 2.1, 3.9, 1.6]] # 新数据，需要和训练集的特征列一致
    my_data = scale.transform(my_data) # 对新数据进行标准化
    my_pred = model.predict(my_data) # 对新数据进行预测
    print(f'新数据预测结果：{my_pred}')
    # 查看上述数据集，每种分类的预测概率
    my_pred_proba = model.predict_proba(my_data)
    print(f'新数据预测概率：{my_pred_proba}')

    # 4.7 模型评估
    #4.7.1 直接评分 基于 训练集的特征 和 训练集的标签
    print(f'准确率: {model.score(x_train, y_train)}')

    # 4.7.2 基于 测试集的标签 和 预测结果
    print(f'准确率: {accuracy_score(y_test, y_pred)}')

# 5、寻找最优超参
def dm05_find_best_k():
    # 5.1 加载鸢尾花数据集
    iris_data = load_iris()

    # 5.2 数据的预处理：从150个特征和标签中，按照8：2的比例划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=23)
    # 5.3 特征提取,由于原数据只有4个特征列，不需要特征提取
    # 5.4 数据标准化, 由于原数据的4列特征差值不大，不需要标准化，但是加入预处理代码更加完善
    scale = StandardScaler()

    # fit 是计算均值和方差，transform 是进行标准化
    x_train = scale.fit_transform(x_train)
    # 把训练集的均值、方差用到测试集，而不是重新计算，所以不加fit   
    x_test = scale.transform(x_test)

    # 5.5 模型训练 与 寻找最优超参
    model = KNeighborsClassifier()
    param_dict = {'n_neighbors': [i for i in range(1, 11)]} # 超参字典，n_neighbors取值范围1-10

    # 创建 GridSearchCV对象
    # 三个参数：要计算最优超参的模型、该模型超参可能出现的值、交叉验证的折数 每个超参组合都要进行4次交叉验证，一共4*10=40次
    # 返回值：处理后的 最优超参模型对象
    model = GridSearchCV(model, param_dict, cv=4)

    # 模型训练动作
    model.fit(x_train, y_train) # x_train是训练集的特征，y_train是训练集的标签

    # 打印最优超参组合
    print(f'最优评分：{model.best_score_}')
    print(f'最优超参组合：{model.best_params_}')
    print(f'最优模型：{model.best_estimator_}')
    # print(f'具体的交叉验证结果：{model.cv_results_}')

    """
    上面得到的结果：
    最优评分：0.9583333333333334
    最优超参组合：{'n_neighbors': 7}
    最优模型：KNeighborsClassifier(n_neighbors=7)
    """
    # 模型评估
    # 对最优超参进行验证
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # 计算准确率
    print(f'准确率：{accuracy_score(y_test, y_pred)}')


# 5、测试
if __name__ == '__main__':
    # dm01_loadIris()
    # dm02_drawIris()
    # dm03_splitData()
    # dm04_iris_evaluate_test()
    dm05_find_best_k()
```



## 6.2 手写数字识别——不太严谨

### 6.2.1 题目分析

**需求：**从数万个手写图像的数据集中正确识别数字

**数据来源：**微信 黑马程序员 小程序可以领取相应课程的数据，不知道csdn怎么上传文件。。

**数据介绍：**

数据文件为一个.csv文件和一张png，包含0-9的手绘数字的灰度图像。

每个图像高28像素，宽28像素，一共784个像素点。像素点取值为[0, 255]。一共785列，784列特征+1列标签

特征名称均有pixel前缀，后面数字[0, 783] 代表像素号， 如pixel666

### 6.2.2 代码实现

#### 6.2.2.1 导入工具类

```python
# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import seaborn as sns # 高级绘图库，用来画统计图、散点图、热力图，方便观察数据分布。
import pandas as pd # 用来处理数据，方便数据清洗、数据筛选、数据转换等操作。
import matplotlib.pyplot as plt # 用来画图，方便观察数据分布、模型预测结果等。
from sklearn.model_selection import train_test_split, GridSearchCV # 划分训练集和测试集 GridSearchCV寻找最优超参与网格搜索
from sklearn.preprocessing import StandardScaler # 数据标准化
from sklearn.neighbors import KNeighborsClassifier #导入 KNN 分类模型
from sklearn.metrics import accuracy_score # 模型评估，计算模型预测的准确率
from collections import Counter # 计算每个类别的数量
import joblib # 保存模型

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
```

#### 6.2.2.2 导入数据与绘图

定义函数，接收用户传入的索引idx，展示对应行数据 对应的图片（练习，实际不需要）

代码如下：

```python
# 1.定义函数，接收用户传入的索引idx，展示对应行数据 对应的图片
def show_digit(idx):
    # 1.读取数据集
    df = pd.read_csv('F:\AllTest\ML\KNN\data\手写数字识别.csv') # 读取后是一个df对象
    # print(df) # 42000行，785列

    # 2.判断索引是否越界
    if idx < 0 or idx > len(df) - 1:
        print('索引越界')
        return
    
    # 3.获取对应行数据
    x_train = df.iloc[:, 1:] # 获取第idx行，第1列到第784列的数据
    y_train = df.iloc[:, 0] # 获取第idx行，第0列的数据

    # 4.查看答案数字
    print(f'该图片对应的数字是：{y_train.iloc[idx]}')

    # 5.展示对应图片
    # print(x_train.iloc[idx].values) #此时是一个784列的一维数组，需要转成28*28的二维数组

    # 6.将一维数组转成二维数组
    x_train = x_train.iloc[idx].values.reshape(28, 28) # 将一维数组转成二维数组，28*28
    print(x_train) # 28 * 28的像素点

    # 7. 绘制灰度图
    plt.imshow(x_train, cmap='gray') # cmap='gray'表示灰度图
    plt.axis('off') # 不显示坐标轴
    plt.show() # 显示图片

# 4.测试
if __name__ == '__main__':
    show_digit(1)
```

#### 6.2.2.3 模型训练

模型训练，模型评估，保存模型

代码如下：

```python
# 2.定义函数，训练模型  ，保存训练好的模型
def train_model():
    # 1.读取数据集
    df = pd.read_csv('F:\AllTest\ML\KNN\data\手写数字识别.csv')

    # 2.划分数据集
    # 拆分出特征列
    x = df.iloc[:, 1:] # 获取第idx行，第1列到第784列的数据
    # 拆分出标签列
    y = df.iloc[:, 0]  # 获取第idx行，第0列的数据

    # 归一化 x = (x - 0) / (255 - 0) 化简就是 x = x / 255
    x = x / 255 # 将数据归一化到0-1之间

    # 3.划分训练集和测试集
    print(f'先查看所有标签的分布情况: {Counter(y)}')
    # 参1：特征列 参2：标签列. 参3：测试集比例. 参4：随机种子. 参5：参考y值进行抽取，保持数据均衡 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y) 

    # 4.模型训练
    # 创建模型对象
    model = KNeighborsClassifier(n_neighbors=3) # n_neighbors=3表示3个邻居

    # 训练模型
    model.fit(x_train, y_train)

    # 5.模型评估
    print(f'准确率: {model.score(x_test, y_test)}')
    print(f'准确率:{accuracy_score(y_test, model.predict(x_test))}')

    # 6.保存模型
    # 参1：模型 参2：保存路径       pickle文件：Python(Pandas)独有的文件格式
    joblib.dump(model, 'F:\AllTest\ML\KNN\model\手写数字识别.pkl') 

# 4.测试
if __name__ == '__main__':
    # show_digit(1)
    train_model()
```

#### 6.2.2.4 模型预测

模型训练好之后，使用训练好的模型，对已有图片进行识别

代码如下：

```python
# 3.定义函数，加载模型，进行识别
def use_model():
    
    # 1.加载图片，把图片变成 数字数组
    x = plt.imread('F:\AllTest\ML\KNN\data\demo.png')

    # 2.绘制灰度图
    # plt.imshow(x, cmap='gray')
    # plt.show()

    # 3.加载模型 读入上一回训练好的模型
    model = joblib.load('F:\AllTest\ML\KNN\model\手写数字识别.pkl')

    # 4.数据集转换，读入的数据是28*28的灰度图，需要转成1*784的数字数组
    x = x.reshape(1, -1) # x.reshape(1, 784) 都可以
    # 记得归一化, 因为训练模型的时候使用了归一化
    # x = x / 255

    # 5.模型预测
    y_pred = model.predict(x)
    print(f'预测结果为：{y_pred}')


# 4.测试
if __name__ == '__main__':
    # show_digit(1)
    # train_model()
    use_model()
```

#### 6.2.2.5 完整代码

需要先调用train_model函数训练并保存模型，再用use_model函数进行模型预测。

完整代码如下：

```python
# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花数据集
import seaborn as sns # 高级绘图库，用来画统计图、散点图、热力图，方便观察数据分布。
import pandas as pd # 用来处理数据，方便数据清洗、数据筛选、数据转换等操作。
import matplotlib.pyplot as plt # 用来画图，方便观察数据分布、模型预测结果等。
from sklearn.model_selection import train_test_split, GridSearchCV # 划分训练集和测试集 GridSearchCV寻找最优超参与网格搜索
from sklearn.preprocessing import StandardScaler # 数据标准化
from sklearn.neighbors import KNeighborsClassifier #导入 KNN 分类模型
from sklearn.metrics import accuracy_score # 模型评估，计算模型预测的准确率
from collections import Counter # 计算每个类别的数量
import joblib # 保存模型


#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1.定义函数，接收用户传入的索引idx，展示对应行数据 对应的图片
def show_digit(idx):
    # 1.读取数据集
    df = pd.read_csv('F:\AllTest\ML\KNN\data\手写数字识别.csv') # 读取后是一个df对象
    # print(df) # 42000行，785列

    # 2.判断索引是否越界
    if idx < 0 or idx > len(df) - 1:
        print('索引越界')
        return
    
    # 3.获取对应行数据
    x_train = df.iloc[:, 1:] # 获取第idx行，第1列到第784列的数据
    y_train = df.iloc[:, 0] # 获取第idx行，第0列的数据

    # 4.查看答案数字
    print(f'该图片对应的数字是：{y_train.iloc[idx]}')

    # 5.展示对应图片
    # print(x_train.iloc[idx].values) #此时是一个784列的一维数组，需要转成28*28的二维数组

    # 6.将一维数组转成二维数组
    x_train = x_train.iloc[idx].values.reshape(28, 28) # 将一维数组转成二维数组，28*28
    print(x_train) # 28 * 28的像素点

    # 7. 绘制灰度图
    plt.imshow(x_train, cmap='gray') # cmap='gray'表示灰度图
    plt.axis('off') # 不显示坐标轴
    plt.show() # 显示图片


# 2.定义函数，训练模型  ，保存训练好的模型
def train_model():
    # 1.读取数据集
    df = pd.read_csv('F:\AllTest\ML\KNN\data\手写数字识别.csv')

    # 2.划分数据集
    # 拆分出特征列
    x = df.iloc[:, 1:] # 获取第idx行，第1列到第784列的数据
    # 拆分出标签列
    y = df.iloc[:, 0]  # 获取第idx行，第0列的数据

    # 归一化 x = (x - 0) / (255 - 0) 化简就是 x = x / 255
    x = x / 255 # 将数据归一化到0-1之间

    # 3.划分训练集和测试集
    print(f'先查看所有标签的分布情况: {Counter(y)}')
    # 参1：特征列 参2：标签列. 参3：测试集比例. 参4：随机种子. 参5：参考y值进行抽取，保持数据均衡 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y) 

    # 4.模型训练
    # 创建模型对象
    model = KNeighborsClassifier(n_neighbors=3) # n_neighbors=3表示3个邻居

    # 训练模型
    model.fit(x_train, y_train)

    # 5.模型评估
    print(f'准确率: {model.score(x_test, y_test)}')
    print(f'准确率:{accuracy_score(y_test, model.predict(x_test))}')

    # 6.保存模型
    # 参1：模型 参2：保存路径       pickle文件：Python(Pandas)独有的文件格式
    joblib.dump(model, 'F:\AllTest\ML\KNN\model\手写数字识别.pkl') 

# 3.定义函数，加载模型，进行识别
def use_model():
    
    # 1.加载图片，把图片变成 数字数组
    x = plt.imread('F:\AllTest\ML\KNN\data\demo.png')

    # 2.绘制灰度图
    # plt.imshow(x, cmap='gray')
    # plt.show()

    # 3.加载模型 读入上一回训练好的模型
    model = joblib.load('F:\AllTest\ML\KNN\model\手写数字识别.pkl')

    # 4.数据集转换，读入的数据是28*28的灰度图，需要转成1*784的数字数组
    x = x.reshape(1, -1) # x.reshape(1, 784) 都可以
    # 记得归一化, 因为训练模型的时候使用了归一化
    # x = x / 255

    # 5.模型预测
    y_pred = model.predict(x)
    print(f'预测结果为：{y_pred}')


# 4.测试
if __name__ == '__main__':
    # show_digit(1)
    # train_model()
    use_model()
```

模型预测有点出入，预测结果不严谨，视频中说是改随机种子就能好，但是可能还是模型和算法选择有些问题。
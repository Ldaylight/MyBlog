# 一、文本预处理

## 1.1 基本概念

文本预处理是 NLP 任务的第一步，目的是将原始文本转化为机器可处理的、结构化的数据。  

原始文本通常包含噪声（HTML 标签、标点、特殊符号、大小写不一致等），且长度不一，无法直接输入模型。  

预处理质量直接影响后续模型性能。

## 1.2 主要环节

文本预处理通常包括以下步骤（顺序可根据任务调整）：

1. **文本清洗**：去除无关字符、HTML 标签、URL、表情符号等。
2. **分词**：将连续文本切分为词或子词单元。
3. **去除停用词**：删除高频但无语义贡献的词（如“的”、“是”、“the”、“is”）。
4. **文本标准化**：统一大小写、数字替换、拼写纠正等。
5. **词干提取 / 词形还原**：将词还原为词根或原形（英文常用）。
6. **构建词汇表**：建立词到索引的映射。
7. **序列截断与填充**：统一输入长度。
8. **文本张量化**：将文本转换为数值张量。

---

## 1.3 文本处理的基本方法

### 1.3.1 文本清洗

#### 1.3.1.1 常见操作

- 移除 HTML 标签（使用正则或 BeautifulSoup）
- 移除 URL、邮箱地址
- 移除特殊符号和标点（根据任务保留部分标点）
- 统一小写
- 处理数字（替换为 `<NUM>` 或保留）
- 处理空白字符

#### 1.3.1.2 示例代码（Python）

```python
import re

def clean_text(text):
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除 URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # 移除特殊字符，仅保留字母、数字、空格、常用标点
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    # 统一小写
    text = text.lower()
    # 合并多个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample = "<p>Hello World! Visit https://example.com or email me@example.com</p>"
print(clean_text(sample))
# 输出: hello world! visit or email
```



### 1.3.2 分词

**分词的定义：**分词（Tokenization）是将==连续的文本序列==切分为有意义的==词单元（token）==的过程。  在英文中，词通常由空格或标点分隔，分词相对简单；在中文等无空格语言中，需要借助词典或统计方法进行切分。

**分词的作用：**

- 将文本转化为可处理单元：模型无法直接处理原始字符串，分词后得到词序列，便于后续建立词汇表和索引化。

- 降低序列长度：以词为单位比以字符为单位序列更短，有利于模型捕捉更长距离的依赖。

- 语义表达的基础：词是承载语义的基本单位，分词质量直接影响后续任务（如情感分析、机器翻译）的性能。

- 支持不同粒度的建模：可进行词级、子词级（BPE、WordPiece）或字符级分词，适应不同语言和任务需求。



#### 1.3.2.1 英文分词

英文通常按空格和标点切分，可使用 `nltk.word_tokenize` 或 `spaCy`。

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Natural language processing is fascinating!"
tokens = word_tokenize(text)
print(tokens)
# ['Natural', 'language', 'processing', 'is', 'fascinating', '!']
```



#### 1.3.2.2 中文分词

中文没有空格分隔，常用 `jieba` 分词。

|               模式               |                   核心特点                   |                      优点                      |                         缺点                         |                       适用场景                       |
| :------------------------------: | :------------------------------------------: | :--------------------------------------------: | :--------------------------------------------------: | :--------------------------------------------------: |
|      **精确模式 (Default)**      |       **默认模式**，追求最精确的切分。       |      分词结果准确，能较好地还原文本原意。      |       召回率相对较低，可能会漏掉一些潜在词语。       |    文本分析、情感分析、信息提取等大多数NLP任务。     |
|        **全模式 (Full)**         |            扫描出所有可能的词语。            |                **速度非常快**。                | 会产生大量**冗余**和**歧义**，结果不能精确还原原句。 | 对分词精度要求不高的场景，或作为搜索引擎的粗筛步骤。 |
| **搜索引擎模式 (Search Engine)** |      在精确模式基础上，对长词再次切分。      |      **召回率高**，能建立更细粒度的索引。      |                  结果同样存在冗余。                  |         构建搜索引擎索引，提升搜索的召回率。         |
|     **Paddle模式 (Paddle)**      | 基于**PaddlePaddle深度学习框架**的分词模式。 | 可进行**词性标注**，在新词识别上可能表现更好。 |             需要额外安装PaddlePaddle库。             |      对分词和词性标注有较高要求的复杂NLP任务。       |

示例代码：

```python
import jieba

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

def jiebacut():
    sentence = '我现在要测试jieba分词的三种分词模式：精确模式分词、全模式分词和搜索引擎模型分词'

    # 1 精确模模式：追求最精确的切分，能较好地还原文本原意
    mydata = jieba.lcut(sentence, cut_all=False)
    print('mydata-->', mydata)

    # 2 全模式分词：扫描出所有可能的词语，速度非常快，会产生大量冗余和歧义
    mydata2 = jieba.lcut(sentence, cut_all=True)
    print('mydata2-->', mydata2)

    # 注意1：人工智能全模型分成三个词
    # 注意2：逗号和句号也给分成了词
    # 3 搜索引擎模式：在精确模式基础上，对长词再次切分，提高召回率，结果同样存在冗余
    mydata3 = jieba.lcut_for_search(sentence)
    print('mydata3-->', mydata3)

if __name__ == '__main__':
    jiebacut()
```



##### 1.3.2.2.1 自定义词典

**准备词典文件**：创建一个 `.txt` 文本文件（**必须保存为 `UTF-8` 编码**）。文件格式为每行一个词，词与词频、词性之间用**空格**隔开。

**格式**：`词语 词频(可省略) 词性(可省略)`

**示例** (`udict.txt`)：

```text
自然语言处理 100 n   # 添加一个名词，词频为100
机器学习             # 只添加词语，词频和词性由jieba自动处理
北京 10 ns           # 添加一个地名，词频为1
```



#####  1.3.2.2.2 **jieba 分词词性标签速查表（了解）**

| 词性标签 |      含义       |                      说明 / 示例                       |
| :------: | :-------------: | :----------------------------------------------------: |
|  **a**   |     形容词      |         取英语 *adjective* 首字母。如：`美丽`          |
|  **ad**  |     副形词      |         直接作状语的形容词。如：`仅仅`、`共同`         |
|  **ag**  |  形容词性语素   |             形容词性语素。如：`美`、`伟大`             |
|  **an**  |     名形词      |        具有名词功能的形容词。如：`困难`、`秘密`        |
|  **b**   |     区别词      |      取汉字“**别**”的声母。如：`男`、`女`、`公共`      |
|  **c**   |      连词       |   取英语 *conjunction* 首字母。如：`和`、`与`、`或`    |
|  **d**   |      副词       |       取 *adverb* 的第2个字母。如：`很`、`非常`        |
|  **dg**  |     副语素      |               副词性语素。如：`偏`、`齐`               |
|  **e**   |      叹词       |      取英语 *exclamation* 首字母。如：`啊`、`哦`       |
|  **f**   |     方位词      |       取汉字“**方**”。如：`上`、`下`、`里`、`外`       |
|  **g**   |      语素       |  取汉字“**根**”的声母。绝大多数语素可作为合成词的词根  |
|  **h**   |    前接成分     |       取英语 *head* 首字母。如：`阿`、`初`、`老`       |
|  **i**   |      成语       |         取英语 *idiom* 首字母。如：`四面八方`          |
|  **j**   |    简称略语     |        取汉字“**简**”的声母。如：`三中`、`全总`        |
|  **k**   |    后接成分     |               如：`儿`、`头`、`性`、`者`               |
|  **l**   |     习用语      |            取“**临**”的声母。如：`综上所述`            |
|  **m**   |      数词       |       取 *numeral* 的第3个字母。如：`一`、`第一`       |
|  **n**   |      名词       |        取英语 *noun* 首字母。如：`桌子`、`电脑`        |
|  **ng**  |   名词性语素    |                     如：`民`、`金`                     |
|  **nr**  |      人名       |  名词代码 `n` 和“人(*ren*)”的声母并在一起。如：`张三`  |
|  **ns**  |      地名       |   名词代码 `n` 和处所词代码 `s` 并在一起。如：`北京`   |
|  **nt**  |   机构团体名    |        “团(*tuan*)”的声母为 `t`。如：`清华大学`        |
|  **nz**  |    其他专名     |       “专(*zhuan*)”的声母为 `z`。如：`诺贝尔奖`        |
|  **o**   |     拟声词      |    取英语 *onomatopoeia* 首字母。如：`哗啦`、`呼呼`    |
|  **p**   |      介词       |  取英语 *prepositional* 首字母。如：`在`、`对`、`向`   |
|  **q**   |      量词       |     取英语 *quantity* 首字母。如：`个`、`条`、`次`     |
|  **r**   |      代词       |     取 *pronoun* 的第2个字母。如：`我`、`你`、`他`     |
|  **rg**  |   代词性语素    |                        如：`何`                        |
|  **rr**  |    人称代词     |                   如：`我们`、`大家`                   |
|  **rz**  |    指示代词     |                 如：`这`、`那`、`这里`                 |
|  **s**   |     处所词      |       取英语 *space* 首字母。如：`郊外`、`远处`        |
|  **t**   |     时间词      |        取英语 *time* 首字母。如：`今天`、`明年`        |
|  **tg**  |     时语素      |              时间词性语素。如：`世`、`纪`              |
|  **u**   |      助词       |  取英语 *auxiliary*。如：`的`、`地`、`得`、`着`、`了`  |
|  **ud**  | 结构助词 **得** |                 如：`走得快` 中的“得”                  |
|  **ug**  |    时态助词     |                  如：`走过` 中的“过”                   |
|  **uj**  | 结构助词 **的** |                 如：`我的书` 中的“的”                  |
|  **ul**  | 时态助词 **了** |                 如：`吃了饭` 中的“了”                  |
|  **uv**  | 结构助词 **地** |                如：`慢慢地走` 中的“地”                 |
|  **uz**  | 时态助词 **着** |                 如：`看着书` 中的“着”                  |
|  **v**   |      动词       |          取英语 *verb* 首字母。如：`吃`、`跑`          |
|  **vd**  |     副动词      |          直接作状语的动词。如：`亲临`、`强行`          |
|  **vg**  |   动词性语素    |                     如：`走`、`跑`                     |
|  **vi**  |   不及物动词    |                   如：`睡觉`、`死亡`                   |
|  **vn**  |     名动词      |     具有名词功能的动词。如：`研究`、`分析`、`调查`     |
|  **w**   |    标点符号     |                  如：`。`、`，`、`！`                  |
|  **x**   |    非语素字     |                 通常代表未知数、符号等                 |
|  **y**   |     语气词      |       取汉字“**语**”的声母。如：`吗`、`呢`、`吧`       |
|  **z**   |     状态词      | 取汉字“**状**”的声母的前一个字母。如：`红彤彤`、`雪白` |

示例代码：

```python
import jieba

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 适用于：稍微生僻的词或者特殊要求的词组
def jieba_userdict():
    sentence = '我十分热爱学习，我要成为人工智能高手'

    # 加载自定义词典
    jieba.load_userdict('./data/userdict.txt')
    
    mydata = jieba.lcut(sentence, cut_all=False)
    print('mydata-->', mydata)

if __name__ == '__main__':
   jieba_userdict()
```



### 1.3.3 去除停用词

停用词表可自定义或使用现成库（如 `nltk.corpus.stopwords`、`jieba.analyse` 或中文停用词表）。

```python
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

tokens = ['natural', 'language', 'processing', 'is', 'fascinating']
filtered = [w for w in tokens if w.lower() not in stop_words]
print(filtered)
# ['natural', 'language', 'processing', 'fascinating']
```



### 1.3.4 文本标准化

#### 1.3.4.1 大小写统一

通常转为小写，减少词汇表大小。

#### 1.3.4.2 数字处理

将所有数字替换为特殊 token，或保留原样。

```python
import re
text = "I have 10 apples and 5 oranges."
text = re.sub(r'\d+', '<NUM>', text)
print(text)
# I have <NUM> apples and <NUM> oranges.
```



### 1.3.5 词干提取与词形还原

#### 1.3.5.1 词干提取（Stemming）

基于规则删除词尾，速度快但可能产生非词。

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words = ["running", "ran", "easily", "happily"]
stemmed = [stemmer.stem(w) for w in words]
print(stemmed)
# ['run', 'ran', 'easili', 'happili']
```



#### 1.3.5.2 词形还原（Lemmatization）

基于词典和词性还原为原形，更准确但速度较慢。

```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
words = ["running", "ran", "better", "geese"]
lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in words]  # 指定词性
print(lemmatized)
# ['run', 'run', 'better', 'goose']
```



### 1.3.6 词性标注

#### 1.3.6.1 POS 简介

**词性标注（Part-of-Speech Tagging）**是为文本中的每个词标注其词性类别，如名词（NN）、动词（VB）、形容词（JJ）等。
常用于：

- **辅助词形还原**：不同词性可能对应不同还原结果（如“better”作为形容词原形是“good”，作为副词原形是“well”）。
- **特征工程**：将词性作为额外特征输入模型。
- **句法分析**：作为依存句法分析等任务的基础。

#### 1.3.6.2 常用工具

- **NLTK**：`pos_tag` 函数，基于 Penn Treebank 标注集。
- **spaCy**：预训练模型包含词性标注组件。
- **jieba**：支持中文词性标注（如 `jieba.posseg`）。

#### 1.3.6.3 代码示例（英文）

```python
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
print(pos_tags)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'),
#  ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]
```

#### 1.3.6.4 代码示例（中文，使用 jieba）

```python
import jieba.posseg as pseg

text = "我爱自然语言处理"
words = pseg.lcut(text)
for word, flag in words:
    print(word, flag)
# 我 r
# 爱 v
# 自然语言 l
# 处理 v
```

#### 1.3.6.5 在预处理中使用 POS 的注意事项

- POS 标注需要额外模型，增加计算开销。
- 标注粒度需与任务匹配（如是否需要细粒度词性）。
- 对于中文，词性标注依赖于分词结果，需先正确分词。

### 1.3.7 命名实体识别NER

#### 1.3.7.1 NER 简介

**命名实体识别（Named Entity Recognition, NER）**是从文本中识别出具有特定意义的实体，如人名、地名、机构名、时间、日期、货币等。

在以下场景中可发挥重要作用：

- **数据脱敏**：将人名、电话号码、身份证号等替换为特殊 token，保护隐私。
- **特征提取**：将实体类型作为额外特征加入模型。
- **辅助分词**：识别复合实体（如“New York”）避免被错误切分。
- **文本标准化**：统一实体表示（如“USA”和“美国”都映射为 `<COUNTRY>`）。

#### 1.3.7.2 常用工具

- **spaCy**：支持多语言，预训练模型包含 NER 组件。
- **Hugging Face Transformers**：使用 pipeline 进行 NER。
- **Stanford NER**：Java 实现，提供多种模型。
- **jieba** 结合自定义词典可实现简单实体识别。

#### 1.3.7.3 代码示例（使用 spaCy）

```python
import spacy

# 加载英文模型（需先安装：python -m spacy download en_core_web_sm）
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is planning to open a new store in New York next month."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
# Apple Inc. ORG
# New York GPE
# next month DATE
```

#### 1.3.7.4 使用 NER 进行实体替换（脱敏）

```python
def replace_entities(text, nlp, mapping={'PERSON': '<PERSON>', 'GPE': '<LOCATION>', 'ORG': '<ORG>'}):
    doc = nlp(text)
    new_tokens = []
    for token in doc:
        if token.ent_type_ in mapping:
            new_tokens.append(mapping[token.ent_type_])
        else:
            new_tokens.append(token.text)
    return ' '.join(new_tokens)

text = "John works at Google in New York."
print(replace_entities(text, nlp))
# <PERSON> works at <ORG> in <LOCATION> .
```

#### 1.3.7.5 注意事项

- NER 本身需要模型，会增加预处理时间和计算成本。
- 预训练 NER 模型可能对特定领域（如医疗、法律）识别效果不佳，需要微调。
- 在预处理阶段使用 NER 时，通常将识别结果作为特征或用于替换，而不是直接删除实体。



### 1.3.8 构建词汇表

将文本转换为索引序列，需要建立词到索引的映射。通常包括特殊 token：`<pad>`（填充）、`<unk>`（未知词）、`<bos>`（句首）、`<eos>`（句尾）。

```python
from collections import Counter

def build_vocab(texts, max_size=10000, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    # 按频率排序
    sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # 保留频率 >= min_freq 的前 max_size 个词
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, freq in sorted_words:
        if freq >= min_freq and len(vocab) < max_size:
            vocab[word] = len(vocab)
    return vocab

texts = ["this is a sample", "this is another example", "example text"]
vocab = build_vocab(texts)
print(vocab)
# {'<pad>': 0, '<unk>': 1, 'this': 2, 'is': 3, 'example': 4, 'a': 5, 'sample': 6, 'another': 7, 'text': 8}
```



### 1.3.9 序列截断与填充

统一长度，通常使用 `pad_sequence` 或自定义函数。

```python
import torch
from torch.nn.utils.rnn import pad_sequence

# 假设已经有索引序列
seqs = [torch.tensor([2,3,4]), torch.tensor([2,3,5,6])]
padded = pad_sequence(seqs, batch_first=True, padding_value=0)
print(padded)
# tensor([[2, 3, 4, 0],
#         [2, 3, 5, 6]])
```



## 1.4 文本张量的表示方法

文本是非结构化数据，文本张量化是将文本转换为数值张量，以便输入神经网络。表示方法的选择直接影响模型性能。常用方法包括：

- **One-hot 编码**：简单、稀疏、高维。
- **词袋模型（BoW）**：统计词频，忽略词序。
- **TF-IDF**：考虑词的重要性。
- **Word2Vec**：学习低维稠密词向量，捕获语义信息。
- **词嵌入（Word Embedding）**：神经网络中的可学习嵌入层，可与任务联合训练或加载预训练向量。

### 1.4.1 One-hot 编码

**One-hot 编码**是将分类变量转化为数字格式的常用方法，对于一个具有n个不同类别的分类变量，将其表示为n维的向量，只有对应位置为 1，其余为 0

点开`Anaconda Prompt`，切换到`nlpbase_CPU`，安装一下`tensorflow`

```bash
pip install tensorflow==2.19.0
```

#### 1.4.1.1 示例代码

```python
import jieba
import tensorflow as tf # 导入keras中的 词汇映射器Tokenizer
import joblib # 保存和加载模型

def dm01_onehot_gen():

    # 1 准备语料 vocabs
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "许嵩", "邓紫棋"}
              
    # 2 实例化词汇映射器Tokenizer, 使用映射器拟合现有文本数据(内部生成 index_word word_index)
    mytokenizer = tf.keras.preprocessing.text.Tokenizer()

    # 通过词汇映射器 在语料库上进行训练
    mytokenizer.fit_on_texts(vocabs)

    # 3 对每个人进行one-hot编码
    for vocab in vocabs:

        # 创建长度 和 词汇映射器一样长的列表，初始值为0
        zero_list = [0] * len(vocabs)

        # 获取当前词在词汇映射器中的索引，索引从1开始，所以要-1
        idx = mytokenizer.word_index[vocab] - 1

        # 修改对应位置的元素为1，完成one-hot编码
        zero_list[idx] = 1
        print(vocab, '的onehot编码是', zero_list)

    # 4 使用joblib工具保存映射器 joblib.dump()
    joblib.dump(mytokenizer, './NLPBaseProject/model/onehot_tokenizer.pkl')
    print('保存完成')

    # 字典没有顺序 onehot编码没有顺序 []-有序 {}-无序 区别
    print(mytokenizer.word_index)
    print(mytokenizer.index_word)

def dm02_use_one_hot():

    # 1 加载已保存的词汇映射器
    mytokenizer = joblib.load('./NLPBaseProject/model/onehot_tokenizer.pkl')

    # 2 编码token为"李宗盛" 查询单词idx 赋值 zero_list，生成onehot
    token = "李宗盛"

    # 创建长度 和 词汇映射器一样长的列表，初始值为0
    zero_list = [0] * len(mytokenizer.word_index)

    # 获取当前词在词汇映射器中的索引，索引从1开始，所以要-1
    idx = mytokenizer.word_index[token] - 1

    # 修改对应位置的元素为1，完成one-hot编码
    zero_list[idx] = 1

    print(token, '的onehot编码是', zero_list)

if __name__ == '__main__':
    # dm01_onehot_gen()
    dm02_use_one_hot()
```

#### 1.4.1.2 优缺点

**优点**：

- 实现简单，直观。
- 适用于小型词汇表或作为基线。

**缺点**：

- 向量维度等于词汇表大小，通常非常大（数十万），导致维度灾难。
- 向量极度稀疏，**属于稀疏词向量表示**，存储和计算效率低。
- 无法表达词之间的语义相似性（所有词之间距离相同）。
- 未利用词频等信息。

通常不直接用于深度学习，但可用于简单分类。



### 1.4.2 Word2Vec

#### 1.4.2.1 基本原理

**Word2Vec** 是一种将单词转换为词向量的方法，通过神经网络在大量文本上训练，将词映射到低维稠密向量空间，用深度学习的网络权重参数表示词向量

是在无监督的语料上，构建了一个有监督的任务

核心思想：**分布式假设**——出现在相似上下文中的词具有相似含义。

#### 1.4.2.2 两种模型结构

##### 1.4.2.2.1 CBOW（Continuous Bag-of-Words）

- 根据上下文词预测中心词。
- 输入：上下文词的 one-hot 向量（或索引），输出：中心词的概率分布。 

##### 1.4.2.2.2 Skip-gram

- 根据中心词预测上下文词。
- 输入：中心词的 one-hot 向量，输出：上下文词的概率分布。

#### 1.4.2.3 训练技巧

- **负采样（Negative Sampling）**：将多分类问题转化为二分类问题，随机采样负样本，提高训练效率。
- **层次 Softmax（Hierarchical Softmax）**：使用 Huffman 树降低计算复杂度。

#### 1.4.2.4 代码示例

先在沙箱中安装词向量训练工具包`FastText`，具有文本分类、训练词向量两大功能

点开`Anaconda Prompt`，切换到`nlpbase_CPU`，安装一下`FastText`

```bash
pip install fasttext-wheel
```

示例代码：

```python
import fasttext

def train_save():

    # 1. 直接开始训练，以无监督的方式运行
    my_model = fasttext.train_unsupervised('./NLPBaseProject/data/wh02ad')

    # 2. 保存模型为 二进制文件， 后续可以通过fasttext.load_model() 加载``
    my_model.save_model('./NLPBaseProject/model/wh02ad_fil9.bin')
    print('训练完成')

# 加载模型并且进行预测
def dm_get_word_vector():
    # 加载模型
    my_model = fasttext.load_model('./NLPBaseProject/model/wh02ad_fil9.bin')

    # 获取某个词的词向量表示
    results = my_model.get_word_vector('the')

# 查看单词的相似度->模型的效果检验
def get_similar_words():
    # 加载模型
    my_model = fasttext.load_model('./NLPBaseProject/model/wh02ad_fil9.bin')

    # 获取词的近义词, 默认是10个 用于检验模型的语义的理解能力
    # 返回格式为 [(相似分数, 近义词), (相似分数, 近义词), (相似分数, 近义词), ...]
    results = my_model.get_nearest_neighbors('dog')

    print(f'results: {results}')

# 实现模型的超参数设定
def set_hyper_params():

    # 手动调整参数
    my_model = fasttext.train_unsupervised(
        './NLPBaseProject/data/wh02ad',     # 训练数据的路径
        model = 'cbow',                     # 词模型类型     
        dim = 50,                           # 词向量的维度
        epoch = 5,                          # 迭代次数
        lr = 0.01,                          # 学习率
        thread = 10,                        # 线程数 
    )

    # 保存模型为 二进制文件， 后续可以通过fasttext.load_model() 加载``
    my_model.save_model('./NLPBaseProject/model/wh02ad_fil9_new.bin')
    print('训练完成')   

if __name__ == '__main__':
    # train_save()
    # get_similar_words()
    set_hyper_params()
```



### 1.4.3 词嵌入（Word Embedding）

Word Embedding 将词映射为**低维稠密向量**，能捕获语义信息。可通过 Word2Vec、GloVe 预训练获得，或在模型中学习。

准确地说：Word Embedding 是将“**词对应的索引**（整数 ID）”转换成词向量，而不是直接处理原始字符串。

在PyTorch 中通过 `nn.Embedding` 层实现

`nn.Embedding(vocab_size, embedding_dim)`：

- `vocab_size`：词典大小（比如 57 个字母，或者 5 万个单词）。
- `embedding_dim`：你想压缩成的维度（比如 128 或 256）。

它就是一个形状为 `[vocab_size, embedding_dim]` 的**二维权重矩阵**



作用：

**1. 维度大幅降低（稀疏 -> 稠密）**：极大减少了模型参数，防止过拟合。

**2. 语义表征（近义词聚集在空间角落）**：Embedding 最伟大的地方在于，它在反向传播训练中，会**根据上下文自动调整权重矩阵里的数字**。

**3. 迁移学习（预训练词向量）**：你不需要每次都从头训练。在工业界，我们通常直接下载 Google 在 1000 亿文本上训好的 `word2vec` 或 `GloVe` 向量，直接拿来初始化你的 Embedding 层，模型效果瞬间起飞（这叫迁移学习）。



nn.Embedding的调用只是一个查表的操作，一开始创建的二维权重矩阵是随机的，数据存储在`self.embedding.weight`中，根据你给出的索引，给出对应的词向量。再反向传播后参数就会进行更新。

nn.Embedding层使用起来更方便就1步。直接嵌入到神经网络中，更易使用，为了让 RNN 处理得舒服，我们**故意**把嵌入维度`embedding_dim`设置得和 RNN 隐藏层维度`hidden_size`一样大。这样 Embedding 输出的向量可以直接作为 RNN 的输入，无需再做线性变换对齐维度。



#### 1.4.3.2 使用tensorboard可视化嵌入的词向量

**需求：**

sentence1 = '我现在要测试jieba分词的三种分词模式：精确模式分词、全模式分词和搜索引擎模型分词‘

 sentence2 = "我爱自然语言处理” 

1、请进行分词，并完成文本数值化，数值张量化处理 

2、可视化展示词向量

**代码如下：**

```python
import torch                                        # 深度学习框架，封装了和张量相关的操作
import tensorflow as tf                             # 导入keras中的 词汇映射器Tokenizer
from torch.utils.tensorboard import SummaryWriter   # 可视化词向量
import jieba                                        # 中文分词
import torch.nn as nn                               # 神经网络模块

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

def embedding_show():

    # 1. 定义待处理的文本
    sentence1 = '我现在要测试jieba分词的三种分词模式：精确模式分词、全模式分词和搜索引擎模型分词'
    sentence2 = "我爱自然语言处理"

    # 2. 把上述的两句话, 封装成 列表. 此时为两个字符串列表
    sentences = [sentence1, sentence2]

    # 3. 使用jieba进行分词
    # 3.1 记录 分词后的词语列表
    word_list = []

    # 3.2 遍历句子
    for sentence in sentences:
        # 3.3 使用jieba进行分词处理, 并将分词结果添加到 word_list 中 此时为[[句子1分词], [句子2分词]]
        word_list.append(jieba.lcut(sentence))

    '''
    [['我', '现在', '要', '测试', 'jieba', '分词', '的', '三种', '分', '词模式', '：', '精确', '模式', '分词', '、', '全', '模式', '分词', '和', '搜索引擎', '模型', '分词'], 
    ['我', '爱', '自然语言', '处理']]
    '''
    # 3.4 打印分词结果.
    print(f'分词结果: {word_list}')

    # 4.构建词汇表, 进行 文本数值化(词向量)
    # 4.1 初始化词汇映射器
    my_tokenizer = tf.keras.preprocessing.text.Tokenizer()

    # 4.2 拟合训练数据, 统计词频, 并构建: 词汇表
    my_tokenizer.fit_on_texts(word_list)

    # 4.3 查看 词 和 索引的映射关系.
    print(f'词和索引的映射关系: {my_tokenizer.word_index}')
    '''
    {'分词': 1, '我': 2, '模式': 3, '现在': 4, '要': 5, '测试': 6, 'jieba': 7, '的': 8, '三种': 9, '分': 10, '词模式': 11, '：': 12, '精确': 13, '、': 14, '全': 15, '和': 16, '搜索引擎': 17, '模型': 18, '爱': 19, '自然语言': 20, '处理': 21}
    '''

    # 4.4 获取去重后的 所有词汇列表
    my_token_list = my_tokenizer.word_index.values()
    print(my_token_list)             # dict_values([1, 2, 3,..., 21])

    # 4.5 将分词后的文本 -> 转成 数字序列.
    seq2id = my_tokenizer.texts_to_sequences(word_list)
    print(f'文本转成数字序列: {seq2id}')       
    ''' 
    [[2, 4, 5, 6, 7, 1, 8, 9, 10, 11, 12, 13, 3, 1, 14, 15, 3, 1, 16, 17, 18, 1], 
    [2, 19, 20, 21]]
    '''

    # 5. 创建 词嵌入层, 把文本(对应的数字编号) 转成 词向量.
    # 5.1 创建 词嵌入层对象 参1: 词汇表大小 参2: 词向量的维度
    #因为 Tokenizer 的索引从1开始，而 Embedding 索引从0开始，所以词汇表大小应为 len(word_index)+1
    embed = nn.Embedding(num_embeddings=len(my_tokenizer.word_index) + 1, embedding_dim=8)

    # 5.2 查看 词嵌入层的 权重参数(即: 词向量)
    print(f'embed: {embed.weight.data}')
    print(f'embed.shape: {embed.weight.data.shape}')        # torch.Size([21, 8])  注意：现在大小为 词汇数+1
    print(' =.= ' * 10)

    # 6. 词向量可视化
    # 6.1 创建TensorBoard写入器, 将数据写入到 runs 目录
    my_summary = SummaryWriter(log_dir='./runs')

    # 修改：获取词语列表（按索引顺序），并跳过索引0对应的无用向量
    words = list(my_tokenizer.word_index.keys())                # 按 word_index 的键顺序得到词语列表
    vectors = embed.weight.data[1:]                             # 取索引1~末尾的向量，对应所有真实词语

    # 6.2 将词向量 和 对应的词语 添加到 TensorBoard中
    # 参1: 词向量矩阵, 形状是: (词汇数, 8), 每个词用8个数字表示(8维)
    # 参2: 对应的词语列表, 用来标注每个点.
    my_summary.add_embedding(vectors, metadata=words)

    # 6.3 关闭写入器
    my_summary.close()

    # 7. 查看每个单词对应的词向量.
    for idx in range(1, len(my_tokenizer.word_index) + 1): # idx的范围: [1, 词汇数]

        # 7.1 获取当前单词对应的词向量.
        temp_vector = embed(torch.tensor(idx)) # tensor([ 0.1227,  0.6931,  0.0187,  1.3011,  0.1021,  2.4951,  0.8910, -0.1551], grad_fn=<EmbeddingBackward0>)  

        # 7.2 获取当前索引对应的单词, my_tokenizer.index_word的索引是从1开始的.
        word = my_tokenizer.index_word[idx]
        print(f'单词: {word}, 词向量: {temp_vector.detach().numpy()}')

        # tensorboard --logdir=runs --host 0.0.0.0

if __name__ == '__main__':
    embedding_show()
```

执行代码之后，会在runs目录生成文件，之后在命令行执行

```bash
tensorboard --logdir=runs --host 0.0.0.0
```

执行后在网页打开网址：http://localhost:6006

效果图如下：

![image-20260902182435204](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260902182435204.png)





## 1.5 NLP沙箱

 **沙箱（Sandbox）**：一种隔离的、临时的运行环境，用于安全地测试代码、模型或配置，不影响生产环境或其他用户。

因此，**NLP 沙箱** 是 用于让用户在受限且安全的条件下运行 NLP 相关代码、训练或推理模型。

### 1.5.1 沙箱创建

```bash
conda env list								#查看所有的沙箱
conda create -n nlpbase_CPU python=3.10		#建议使用3.10版本，nlpbase_CPU就是沙箱名
conda activate nlpbase_CPU					#切换沙箱
pip list									#查看当前沙箱装了哪些包
pip install jieba							#安装jieba分词包
```

### 1.5.2 项目关联沙箱

创建一个文件夹命名`NLPBase_Project`

进入vscode，通过`ctrl + shift + P`选择环境

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260901125206515.png" alt="image-20260901125206515" style="zoom: 80%;" />

在终端中，显示`(nlpbase_CPU)`，表示成功

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260901125510047.png" alt="image-20260901125510047" style="zoom:80%;" />



## 1.6 文本数据分析

**文本数据分析**能够有效帮助我们理解数据语料, 快速检查出**语料可能存在的问题**, 指导模型训练过程中一些超参数的选择 

比如：标签Y：分类问题查看标签是否均匀；数据X：数据有没有脏数据、数据长度分布等等

常用的几种文本数据分析方法：标签数量分布、句子长度分布、词频统计与关键词词云



### 1.6.1 标签数量分布

标签数量分布是指数据集中**各个类别标签出现的频率统计**。对于分类任务，了解标签分布有助于**判断数据是否平衡**，是否需要进行重采样

或调整损失函数。  可视化标签分布通常使用条形图或饼图。

#### 1.6.1.1 示例代码

```python
import pandas as pd
import matplotlib.pyplot as plt

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

labels = ['positive', 'negative', 'positive', 'neutral', 'positive', 'negative', 'neutral', 'positive']

# 统计频数并转为 DataFrame（保留类别顺序）
counts = pd.Series(labels).value_counts()
print(counts)

# 准备颜色列表（每个柱子一种颜色）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿，可根据类别数扩展

# 绘制条形图
ax = counts.plot(kind='bar', color=colors)  # 也可用 matplotlib 的 bar
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')

# 强制 x 轴标签水平
plt.xticks(rotation=0)

# 在每个柱顶添加数值标签
for i, v in enumerate(counts):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')

# 自动调整布局
plt.tight_layout()
plt.show()
```

效果如下：

<img src="C:/Users/86176/AppData/Roaming/Typora/typora-user-images/image-20260902192029328.png" alt="image-20260902192029294" style="zoom: 80%;" />

### 1.6.2 句子长度分布

句子长度分布是指数据集中**每个文本样本的 token 数量（或字符数）的统计分布**。它帮助确定合适的序列最大长度，指导后续的**截断与**

**填充**策略。通常使用直方图或箱线图展示长度分布。



#### 1.6.2.1 直方图和密度曲线

可以直观的查看句子长度分布最密集的范围

示例代码：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']     
plt.rcParams['axes.unicode_minus'] = False

# VS Code 终端默认编码不是 UTF-8（非VS Code 可以不用写）
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ---------- 模拟一个较大的句子长度数据集 ----------
# 假设句子长度服从泊松分布（均值为8），取1000个样本，长度范围1~30
np.random.seed(42)
lengths = np.random.poisson(lam=8, size=1000)
lengths = lengths[lengths > 0]          # 剔除长度为0的句子
lengths = lengths[lengths <= 30]        # 截断过长值（可选）
print(f"生成了 {len(lengths)} 个句子的长度，示例：{lengths[:10]}")

# ---------- 图1：直方图 + 密度曲线（叠加） ----------
plt.figure(figsize=(10, 5))
sns.histplot(lengths, bins=range(1, 31), kde=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('句子长度分布（直方图 + 密度曲线）')
plt.xlabel('句子长度（词数）')
plt.ylabel('频数')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------- 图2：单独的密度曲线（KDE） ----------
plt.figure(figsize=(10, 5))
sns.kdeplot(lengths, fill=True, color='orange', linewidth=2)
plt.title('句子长度密度曲线')
plt.xlabel('句子长度（词数）')
plt.ylabel('概率密度')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

效果图如下：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260902195806098.png" alt="image-20260902195806098" style="zoom: 80%;" />

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260902195829204.png" alt="image-20260902195829204" style="zoom: 80%;" />



#### 1.6.2.2 散点图

查看正负样本长度散点图，可以有效定位**异常点出现的位置**，帮助我们更准确进行人工语料审查

示例代码：

```python
def sns_stripplot():
    # 1. 读取训练集 和 测试集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 获取(训练集)数据长度列
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))

    # 3. 获取(测试集)数据长度列
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))

    # 4. 统计正负样本长度的 散点分布
    # 训练集
    # 参1: x轴标签   参2: y轴标签   参3: 数据集    参4: 用于分组的字段
    sns.stripplot(x='label', y='sentence_length', data=train_data, hue='label')
    plt.title('训练集正负样本长度散点分布')
    plt.show()

    # 测试集
    sns.stripplot(x='label', y='sentence_length', data=dev_data, hue='label')
    plt.title('测试集正负样本长度散点分布')
    plt.show()
```

效果图如下：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903131252758.png" alt="image-20260903131252758" style="zoom: 80%;" />



### 1.6.3 单词总数

单词总数通常指整个数据集中所有词的总出现次数（总词频），或去重后的词汇表大小。它反映了数据规模和词汇丰富度。
分析单词总数**有助于确定词汇表大小**、是否出现大量低频词等。

#### 常用方法

- **总词频**：将所有文本分词后累加 token 数量。
- **去重词汇量**：使用 `set` 统计不同词的数量。
- **平均句长**：总词数除以样本数。

#### 示例代码

```python
from collections import Counter

# 假设已分词文本
texts = [
    ['this', 'is', 'a', 'sample'],
    ['this', 'is', 'another', 'example'],
    ['example', 'text']
]

# 展平所有 token
all_tokens = [token for tokens in texts for token in tokens]
total_words = len(all_tokens)
unique_words = len(set(all_tokens))
avg_length = total_words / len(texts)

print(f"总词数: {total_words}")
print(f"去重词汇量: {unique_words}")
print(f"平均句长: {avg_length:.2f}")
```



### 1.6.4 词频和关键词云

词频统计是指计算每个词在数据集中出现的次数，可用于发现高频词、停用词、主题词等。

关键词云是一种可视化方式，将词频以字体大小表示，直观展示文本中的重点词汇。可以用于对当前语料质量进行简单评估 和人工审查

#### 常用方法

- **词频统计**：使用 `collections.Counter` 统计。
- **词云绘制**：使用 `wordcloud.WordCloud` 库，可根据词频生成图片。

#### 示例代码

```python
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 假设已分词文本
texts = [
    ['this', 'is', 'a', 'sample'],
    ['this', 'is', 'another', 'example'],
    ['example', 'text', 'for', 'wordcloud'],
    ['sample', 'text', 'example']
]

# 统计词频
all_tokens = [token for tokens in texts for token in tokens]
word_counts = Counter(all_tokens)
print(word_counts.most_common(5))

# 生成词云
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

效果图如下：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903135928252.png" alt="image-20260903135928252" style="zoom:50%;" />



### 1.6.5 项目实例——

#### 1.6.5.1 项目说明



#### 1.6.5.2 标签数量分布

先查看训练集和测试集的标签数量分布

代码如下：

```python
def label_sns_countplot():

    # 读取训练集 和 测试集
    # 参1: 文件路径.   参2: 列分隔符(csv文件用,  tsv文件用\t)
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')
    # print(train_data.head())

    # 统计训练集标签的 0(负样本) 和 1(正样本) 的数量, 并可视化, 采用: 计数柱状图.
    # 参1: x轴标签,  参2: 数据集,  参3: 用于分组的分类变量,  参4: 是否显示图例(默认为True)
    ax = sns.countplot(x='label', data=train_data, hue='label', legend=False)
    plt.title('train_label')    # 设置标题
    
    # 在柱顶添加数值
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.,
                height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()          # 紧凑布局
    plt.show()

    # 统计测试集标签的 0(负样本) 和 1(正样本) 的数量, 并可视化, 采用: 计数柱状图.
    ax = sns.countplot(x='label', data=dev_data, hue='label', legend=False)
    plt.title('dev_label')
    
    # 在柱顶添加数值
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2.,
                height + 0.1,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()
```

结果图如下：

训练集：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260902195003630.png" alt="image-20260902195003630" style="zoom: 80%;" />

测试集：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260902195116745.png" alt="image-20260902195116745" style="zoom: 80%;" />



#### 1.6.5.3 句子长度分布

查看一下训练集和测试集的句子长度的分布

代码如下：

```python
def len_sns_distplot():

    # 读取训练集 和 测试集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 计算训练集的 每个句子的 长度.
    # map(函数, 可迭代对象) 把 函数 依次应用到 可迭代对象 的每一个元素上
    # lambda x: len(x) 匿名函数，给它一个x，返回 x 的长度 len(x)
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))

    # 绘制训练集的 句子长度分布
    # 图1: 计数柱状图
    sns.countplot(x='sentence_length', data=train_data)
    plt.title('训练集句子长度分布_计数柱状图')
    plt.xticks([])      # 隐藏x轴刻度值
    plt.show()

    # histplot() 直方图
    sns.histplot(x='sentence_length', data=train_data, kde=True)
    plt.title('训练集句子长度分布_密度曲线图')
    plt.show()

    # 4. 计算测试集的 每个句子的 长度.
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))
    # 图1: 计数柱状图
    sns.countplot(x='sentence_length', data=dev_data)
    plt.title('测试集句子长度分布_计数柱状图')
    plt.xticks([])
    plt.show()

    # 图2: 密度曲线图. 
    sns.histplot(x='sentence_length', data=dev_data, kde=True)
    plt.title('测试集句子长度分布_密度曲线图')
    plt.show()
```

效果图如下：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903123057918.png" alt="image-20260903123057918" style="zoom:50%;" />

<img src="C:/Users/86176/AppData/Roaming/Typora/typora-user-images/image-20260903123129074.png" alt="image-20260903123129074" style="zoom: 80%;" />

正负样本散点图如下：

```python
def sns_stripplot():
    # 1. 读取训练集 和 测试集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 获取(训练集)数据长度列
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))

    # 3. 获取(测试集)数据长度列
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))

    # 4. 统计正负样本长度的 散点分布
    # 训练集
    # 参1: x轴标签   参2: y轴标签   参3: 数据集    参4: 用于分组的字段
    sns.stripplot(x='label', y='sentence_length', data=train_data, hue='label')
    plt.title('训练集正负样本长度散点分布')
    plt.show()

    # 测试集
    sns.stripplot(x='label', y='sentence_length', data=dev_data, hue='label')
    plt.title('测试集正负样本长度散点分布')
    plt.show()
```

效果图如下：

训练集：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903131252758.png" alt="image-20260903131252758" style="zoom: 80%;" />

测试集：

<img src="C:/Users/86176/AppData/Roaming/Typora/typora-user-images/image-20260903131441571.png" alt="image-20260903131441535" style="zoom: 80%;" />

#### 1.6.5.4 单词总数

查看jieba分词后，每句话的token数

代码如下：

```python
def get_word_count():
    # 1. 读取训练集 和 测试集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 统计训练集的 词汇总数(去重后的).
    # * 是 Python 的解包操作符,把 map 产生的迭代器中的每一个元素（即多个列表），拆开作为独立的参数
    # chain把多个列表 首尾相连 拼接成一个大的迭代器
    # set 是 集合，自动去重
    train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))
    print(f'训练集共包含不同词汇总数为: {len(train_vocab)}')

    # 3. 统计测试集的 词汇总数(去重后的).
    dev_vocab = set(chain(*map(lambda x: jieba.lcut(x), dev_data['sentence'])))
    print(f'测试集共包含不同词汇总数为: {len(dev_vocab)}')
```



#### 1.6.5.5 词频和词云

对训练集中正负样本的形容词做一个关键词词云，

示例代码：

```python
def word_cloud():
    
    # 场景1: 处理 训练集 -> 正样本
    # 1. 读取训练集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')

    # 2. 处理 训练集的 正样本(label=1) -> 生成词云.
    # 2.1 筛选label=1的样本, 并提取句子列.
    p_train_data = train_data[train_data['label'] == 1]['sentence']
    # 2.2 对每个正样本句子, 提取形容词列表, 并合并为1个完整的 形容词列表.
    p_a_train_vocab = chain(*map(lambda x: get_a_list(x),  p_train_data))
    # 2.3 调用词云函数, 根据形容词列表, 绘制词云.
    get_word_cloud(p_a_train_vocab)

    # 分隔符
    print(' =.= ' * 10)

    # 场景2: 处理 训练集 -> 负样本
    # 2. 处理 训练集的 负样本(label=0) -> 生成词云.
    # 2.1 筛选label=0的样本, 并提取句子列.
    p_train_data = train_data[train_data['label'] == 0]['sentence']
    # 2.2 对每个负样本句子, 提取形容词列表, 并合并为1个完整的 形容词列表.
    p_a_train_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
    # 2.3 调用词云函数, 根据形容词列表, 绘制词云.
    get_word_cloud(p_a_train_vocab)
```



正样本词云：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903142124427.png" alt="image-20260903142124427" style="zoom: 80%;" />

负样本词云：

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903142157545.png" alt="image-20260903142157545" style="zoom: 80%;" />



## 1.7 文本特征处理

### 1.7.1 n-gram 特征

**n-gram** 是指文本中连续的 n 个词（或字符）组成的序列。n-gram 特征通过将相邻词组合起来，捕获局部词序信息，弥补词袋模型忽略词序的不足。n一般不会超过3

uni-gram(1-gram): 把 每个词/字 拆出来

bi-gram(2-gram):  找连续2个词的组合

tri-gram(3-gram): 找连续3个词的组合

#### 常用方法

- **生成 n-gram**：使用 `nltk.ngrams` 或手动滑动窗口生成。
- **特征表示**：将 n-gram 作为额外特征加入 BoW 或 TF-IDF 向量中，可通过 `sklearn.feature_extraction.text.CountVectorizer` 的 `ngram_range` 参数实现。

#### 示例代码

```python
#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# n的值, 一般 n-gram中的n取 2 或者 3, 这里以 2 举例
ngram_range = 2

# 定义函数, 生成 n-gram 特征
def create_ngram(input_list):
    # 1. 通过滑动窗口, 获取 n-gram特征
    # i = 0 -> input_list[0:] -> [1, 3, 2, 1, 5, 3]
    # i = 1 -> input_list[1:] -> [3, 2, 1, 5, 3]
    sliced_lists = [input_list[i:] for i in range(ngram_range)]

    # 2. 使用zip()函数, 对切片列表进行组合.
    ngram_tuples = zip(*sliced_lists)

    # 3. 转换为集合(去重), 虽然本样例中无重复, 但是保证结果唯一.
    return set(ngram_tuples)

# todo 3. 测试代码
if __name__ == '__main__':
    # 1. 定义列表, 记录: 输入数值.
    input_list = [1, 3, 2, 1, 5, 3]

    # 2. 调用函数生成 n-gram 特征
    result = create_ngram(input_list)

    # 3. 输出结果.
    print(result)
```



### 1.7.2 文本长度规范和作用

**文本长度规范**是指将变长文本序列转换为固定长度（或限制在某个范围内），以满足神经网络输入要求。常见的处理包括**截断（truncation）和填充（padding）**。

- **截断**：对超过最大长度的序列进行裁剪，可保留开头、结尾或中间部分。
- **填充**：对不足最大长度的序列补充特殊 token（如 `<pad>`），通常在 batch 内进行动态填充。

#### 常用方法

- **固定长度截断与填充**：事先设定一个最大长度 `cutlen`，对每个序列进行截断或填充。
- **动态填充**：在同一个 batch 内，以该 batch 中最长序列为基准进行填充，减少计算浪费。

#### 示例代码

```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 导包
import tensorflow as tf

# VS Code 终端默认编码不是 UTF-8（非VS Code 可以不用写）
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 定义遍历, 记录: 截断补齐长度参数.
cutlen = 10         # 实际开发中, 根据你的语料库的句子长度分布来自定义.

# 定义函数, 对输入的文本张量进行截断补齐
def padding(x_train):
    # 参1: 待处理的文本张量.
    # 参2: 最大长度.
    # 参3: 截断策略, pre(不要序列前端的元素), post(不要序列后端的元素)
    # 参4: 填充策略,  pre(默认, 从序列前端补齐)   post(从序列后端补齐)
    # return sequence.pad_sequences(x_train, maxlen=cutlen)
    # return sequence.pad_sequences(x_train, maxlen=cutlen, truncating='pre', padding='pre')
    return tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=cutlen, truncating='post', padding='post')

#方法二
# 定义函数, 对输入的文本张量进行截断补齐
def padding_custom(x_train):
    # 1. 定义遍历, 记录: 初始化列表.
    list1 = []
    # 2. 遍历语料库中每个句子
    for sentence in x_train: 
        # 3. 处理超长文本, 截断维度, 保留前cutlen个元素
        if len(sentence) > cutlen:
            # 截断并添加到列表中
            list1.append(sentence[:cutlen])
        # 4. 处理短序列
        else:
            # 4.1 计算需要补齐 0 的量.
            padding_len = cutlen - len(sentence)
            # 4.2 创建补齐的列表, 并添加到列表中.
            list1.append(sentence + [0] * padding_len)

    # 5. 返回处理后的列表.
    return list1

# todo 3. 测试代码.
if __name__ == '__main__':
    # 1.定义遍历, 记录: 文本信息.
    x_train = [
        [1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
        [2, 32, 1, 23, 1]
    ]

    # 2. 调用上述的函数, 实现: 截断, 补齐.
    # result = padding(x_train)
    result = padding_custom(x_train)

    # 3. 打印处理后的内容.
    print(f'result: {result}')
```



## 1.8 文本数据增强

### 1.8.1 回译数据增强法

回译（Back Translation）是一种常用的文本数据增强技术，尤其用于机器翻译和文本分类任务。其基本思想是**将原始文本翻译成另一种**

**语言**，**再将其翻译回原语言**。由于翻译过程中的措辞变化，生成的文本与原文语义相近但表达不同，从而扩充训练数据。

#### 常用方法

- **基于 API 的回译**：调用谷歌翻译、百度翻译等在线翻译 API，进行两次翻译。
- **基于本地模型**：使用预训练的神经机器翻译模型（如 Helsinki-NLP 的 MarianMT）在本地完成翻译，避免网络依赖。
- **注意事项**：回译可能引入噪声，需保证翻译质量；对于领域特定文本，通用翻译模型可能效果不佳。

#### 示例代码（使用 Helsinki-NLP 模型）

```python
from transformers import MarianMTModel, MarianTokenizer

# 加载英→法和法→英模型
src_lang = "en"
tgt_lang = "fr"
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 反向模型
back_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"
back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)
back_model = MarianMTModel.from_pretrained(back_model_name)

def back_translate(text, max_length=128):
    # 英→法
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    translated = model.generate(**inputs)
    french_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    # 法→英
    back_inputs = back_tokenizer(french_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    back_translated = back_model.generate(**back_inputs)
    english_text = back_tokenizer.decode(back_translated[0], skip_special_tokens=True)
    return english_text

# 示例
original = "The quick brown fox jumps over the lazy dog."
augmented = back_translate(original)
print("Original:", original)
print("Augmented:", augmented)
```





# 二、循环神经网络

## 2.1 认识RNN模型

### 2.1.1 RNN模型介绍

**循环神经网络**（Recurrent Neural Network, **RNN**）是一类用于处理**序列数据**的神经网络。与传统前馈神经网络不同，RNN 引入**隐藏状态**（hidden state），使网络能够记住之前时间步的信息，从而对序列的时序动态进行建模。RNN 在每个时间步接收当前输入和前一时刻的隐藏状态，输出当前隐藏状态和（可选的）当前输出，参数在所有时间步共享。

给定输入序列 $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$，RNN 的隐藏状态更新公式为：

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{ih} \mathbf{x}_t + \mathbf{b}_{ih} + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_{hh})
$$

其中 $\mathbf{W}_{ih}, \mathbf{W}_{hh}$ 是权重矩阵，$\mathbf{b}_{ih}, \mathbf{b}_{hh}$ 是偏置，$\tanh$ 是激活函数。

### 2.1.2 RNN模型的作用

RNN 特别适合处理具有时序或序列特性的数据，在 NLP 中有广泛应用：

- **语言建模**：预测下一个词或字符。
- **文本分类**：对整个序列的输出（如最后时间步的隐藏状态）进行分类。
- **序列标注**：词性标注、命名实体识别等，每个时间步输出一个标签。
- **机器翻译**：作为 Seq2Seq 模型的编码器和解码器。
- **文本生成**：逐词生成文本。

RNN 的核心优势是能够处理变长序列，并捕获序列中的依赖关系。

### 2.1.3 RNN模型的分类

根据输入和输出的序列长度关系，RNN 可以划分为以下几种结构：

1. **一对一（1-to-1）**：即普通前馈网络，输入和输出都是固定长度的向量，不涉及时间维度。

2. **一对多（1-to-N）**：输入是单一向量，输出是序列。例如图像描述生成（输入图像特征，输出单词序列）。

   <img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903195542977.png" alt="image-20260903195542977" style="zoom: 80%;" />

3. **多对一（N-to-1）**：输入是序列，输出是单一向量。例如文本分类、情感分析。

   <img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903195517360.png" alt="image-20260903195517360" style="zoom: 80%;" />

4. **多对多（N-to-N）**：输入和输出都是序列，且长度相同。例如序列标注（每个词对应一个标签）。

   <img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903195335583.png" alt="image-20260903195335583" style="zoom: 80%;" />

5. **多对多（N-to-M）**：输入和输出都是序列，但长度可以不同。例如机器翻译，通常使用 Encoder-Decoder 架构（Seq2Seq）。

<img src="https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260903195617909.png" alt="image-20260903195617909" style="zoom: 80%;" />

这些结构可以通过组合 RNN 单元和不同的输入输出方式实现。



按照 内部结构 划分:

传统的RNN：输入层, 隐藏层(词嵌入层, 循环网络层), 输出层

LSTM：遗忘门, 输入门, 细胞状态, 输出门

Bi-LSTM：针对于语料, 从前往后做一次LSTM, 从后往前做一次LSTM, 然后将两个结果拼接起来.

GRU：重置门, 更新门

Bi-GRU：针对于语料, 从前往后做一次GRU, 从后往前做一次GRU, 然后将两个结果拼接起来.

---

## 2.2 传统RNN模型

### 2.2.1 内部结构分析

传统 RNN的内部结构相对简单：一个 RNN 单元在时间步 $t$ 接收输入 $\mathbf{x}_t$ 和上一个时间步的隐藏状态 $\mathbf{h}_{t-1}$，产生新的隐藏状态 $\mathbf{h}_t$。计算过程如下：

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{ih} \mathbf{x}_t + \mathbf{b}_{ih} + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_{hh})
$$

输出 $\mathbf{y}_t$ 通常由隐藏状态经过线性变换得到：

$$
\mathbf{y}_t = \mathbf{W}_{ho} \mathbf{h}_t + \mathbf{b}_o
$$

其中 $\mathbf{W}_{ih} \in \mathbb{R}^{d \times h}$，$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$，$\mathbf{W}_{ho} \in \mathbb{R}^{h \times o}$，$d$ 是输入维度，$h$ 是隐藏维度，$o$ 是输出维度。

**结构特点**：

- 参数共享：所有时间步使用相同的权重矩阵。
- 隐藏状态是网络的“记忆”，传递历史信息。
- 激活函数通常用 $\tanh$ 来引入非线性。

### 2.2.2 RNN模型API

#### 2.2.2.1 nn.Linear 介绍

`nn.Linear` 是 PyTorch 中**最基础、最核心**的神经网络层，它的专业名称叫**全连接层**或**线性层**

`nn.Linear`在后台执行的是一条非常简单的数学公式：

![image-20260909160939592](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260909160939592.png)



`nn.Linear(in_features, out_features, bias=True)`

|       参数名       |     中文含义     |                在本任务中的具体数值                 | 数据类型 |                             作用                             |
| :----------------: | :--------------: | :-------------------------------------------------: | :------: | :----------------------------------------------------------: |
| **`in_features`**  |  **输入特征数**  |      你代码里的 `hidden_size`（比如 **128**）       |  `int`   | 规定**喂进来**的数据最后一维有多大。如果输入是 `[batch, 128]`，这里必须填 `128`。 |
| **`out_features`** |  **输出特征数**  | 你代码里的 `output_size`（即 **18**，代表18个国家） |  `int`   | 规定**吐出去**的数据最后一维有多大。比如你想得到18个分数，这里就填 `18`。 |
|     **`bias`**     | **是否使用偏置** |             默认 `True`（通常保持默认）             |  `bool`  |            决定是否在输出上加一个可学习的常数项。            |

**1.特征空间的升降维（投影）**

这是它最根本的功能。它不改变数据的“批次大小”和“序列长度”，==只改变最后一维的大小==。

- **降维**：比如把 `[batch, 4096]` 压缩成 `[batch, 10]`（用于分类）。
- **升维**：比如把 `[batch, 100]` 扩展成 `[batch, 512]`（用于增强表达）。

**2.分类任务的“决策输出层”（分类头）**

在绝大多数深度学习分类任务中，`nn.Linear` 位于模型的最后一层。它的 `out_features` 等于类别总数。

- 前一层提取“特征”（如 RNN 的隐藏状态），它负责把特征换算成“每个类别的得分（Logits）”。分数最高的那个类，就是模型的预测结果。

 **3.特征提取与组合（线性变换）**

即使不是最后一层，多层 `nn.Linear` 堆叠（中间加上激活函数）可以学习到输入特征之间复杂的非线性组合关系。



当你创建 `nn.Linear(128, 18)` 时，PyTorch 在后台

- **创建了 `self.weight`（权重矩阵）**：形状是 `[18, 128]`，里面塞满了随机初始化的浮点数。
- **创建了 `self.bias`（偏置向量）**：形状是 `[18]`，初始化为 0 或极小值。

真正参与计算的变量是你在前向传播调用线性层而传进来的数据x

如： x --> [1, 128]

经过一次线性变换 x wT + b -->  [1, 128] * [128, 18] + [18] = [1, 18]

示例代码：

```python
import torch
import torch.nn as nn

# 创建线性层：输入128维，输出18维
linear = nn.Linear(128, 18)

# 后台生成的随机矩阵W 和 向量b
print(linear.weight.shape)         # torch.Size([18, 128]) —— 一个二维矩阵
print(linear.bias.shape)           # torch.Size([18])      —— 一个一维向量

# 模拟输入
x = torch.randn(1, 128)            # [1, 128]
y = linear(x)                      # 前向传播
print(y.shape)                     # torch.Size([1, 18])   —— 这是一个二维张量
```



nn.Linear的使用场景

|     场景 / 位置      |       代码示例        |    输入形状    |    输出形状    |                         输出物理含义                         |
| :------------------: | :-------------------: | :------------: | :------------: | :----------------------------------------------------------: |
| **隐藏层（中间层）** | `nn.Linear(128, 256)` | `[batch, 128]` | `[batch, 256]` |                 高维抽象特征，无直接可解释性                 |
|    **分类输出层**    | `nn.Linear(256, 18)`  | `[batch, 256]` | `[batch, 18]`  | **Logits（原始分数）**，每个类别一个分数，有正有负，不是概率 |
|    **回归输出层**    |  `nn.Linear(256, 1)`  | `[batch, 256]` |  `[batch, 1]`  |                预测的连续数值（房价、温度等）                |
|   **注意力打分器**   | `nn.Linear(512, 10)`  | `[batch, 512]` | `[batch, 10]`  |         **注意力分数（未归一化）**，每个源词一个分数         |
| **融合层 / 投影层**  | `nn.Linear(512, 256)` | `[batch, 512]` | `[batch, 256]` |       **融合后的隐藏层特征**，把拼接向量压缩回指定维度       |



#### 2.2.2.2 常用参数对照

|     张量类别      |    维度1（长度/层数）    |    维度2（批量）    |         维度3（特征）         |                         **铁律要求**                         |
| :---------------: | :----------------------: | :-----------------: | :---------------------------: | :----------------------------------------------------------: |
| **输入 `input`**  |  `seq_len`（序列长度）   | `batch`（批次大小） | **`input_size`**（输入特征）  |  **③ 输入特征维** 必须等于 `nn.RNN` 初始化时的 `input_size`  |
| **隐藏 `hidden`** | **`num_layers`**（层数） | `batch`（批次大小） | **`hidden_size`**（隐藏特征） | **① 层数维** 必须等于 `nn.RNN` 初始化时的 `num_layers` **③ 隐藏特征维** 必须等于 `hidden_size` |
| **输出 `output`** |  `seq_len`（序列长度）   | `batch`（批次大小） | **`hidden_size`**（隐藏特征） | **② 序列长度** 与输入 `seq_len` 保持一致 **③ 输出特征维** 等于 `hidden_size` |



#### **2.2.2.3 基础版RNN**

示例代码：

```python
def rnn_for_base():
    # 1. 创建RNN模型.
    # 参1: 词向量维度(输入维度),  参2: 隐藏层维度(输出维度),  参3: 隐藏层层数.
    rnn = nn.RNN(5, 6, 1)

    # 2. 准备输入数据(即: 本次的输入)
    # 参1: 句子长度(sequence_length),  参2: 批次大小(batch_size),  参3: 词向量维度(即:输入维度 input_size)
    input = torch.randn(1, 3, 5)

    # 3. 初始化隐藏层(即: 上一时间步的隐藏状态)
    # 参1: 隐藏层的层数(即: 隐藏层层数 num_layers),  参2: 批次大小(batch_size),  参3: 隐藏层维度(即: 输出维度 hidden_size)
    h0 = torch.randn(1, 3, 6)

    # 4. 运行RNN模型
    # 本次的输出, 本次的隐藏状态 = rnn(本次的输入, 上一时刻的隐藏状态)
    output, hn = rnn(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, output.shape: {output.shape}')        # shape: (1, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')                # shape: (1, 3, 6)
    print(f'rnn模型: {rnn}')                                         # RNN(5, 6)

```



#### 2.2.2.4 修改RNN模型(句子)的长度

修改输入句子长度，本次的输出需要更改

示例代码：

```python
def rnn_for_sequence_len():
    # 1. 创建RNN模型.
    # 参1: 词向量维度(输入维度),  参2: 隐藏层维度(输出维度),  参3: 隐藏层层数.
    rnn = nn.RNN(5, 6, 1)

    # 2. 准备输入数据(即: 本次的输入)
    # 参1: 句子长度(sequence_length),  参2: 批次大小(batch_size),  参3: 词向量维度(即:输入维度 input_size)
    input = torch.randn(20, 3, 5)

    # 3. 初始化隐藏层(即: 上一时间步的隐藏状态)
    # 参1: 隐藏层的层数(即: 隐藏层层数 num_layers),  参2: 批次大小(batch_size),  参3: 隐藏层维度(即: 输出维度 hidden_size)
    h0 = torch.randn(1, 3, 6)

    # 4. 运行RNN模型
    # 本次的输出, 本次的隐藏状态 = rnn(本次的输入, 上一时刻的隐藏状态)
    output, hn = rnn(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, output.shape: {output.shape}')        # shape: (20, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')                # shape: (1, 3, 6)
    print(f'rnn模型: {rnn}')                                         # RNN(5, 6)
```

### 2.2.3 优缺点

**优点**：

- 短序列任务中性能和效果都表现优异。
- 结构简单，计算资源要求低，易于实现。

**缺点**：

- **梯度消失/爆炸**：在长序列上训练时，通过时间反向传播（BPTT）的梯度可能指数级衰减或增长，导致难以学习长期依赖。
- **长期记忆能力有限**：由于梯度消失，传统 RNN 实际上只能记住较短时间步的信息。
- **计算效率低**：序列必须按时间步顺序计算，难以并行化。

---

## 2.3 LSTM模型

### 2.3.1 模型介绍

长短期记忆网络（Long Short-Term Memory, LSTM）是一种特殊的 RNN，专门设计用来解决传统 RNN 的长期依赖问题。LSTM 通过引入**细胞状态**（cell state）和**门控机制**（gating mechanism）来控制信息的流动，能够在较长序列中有效地保留或遗忘信息。

LSTM 的核心思想是让信息可以选择性地通过“门”结构。每个 LSTM 单元包含三个门：遗忘门、输入门、输出门，以及一个细胞状态。

与传统RNN相比，能**有效捕捉长序列**之间的语义关联，**缓解梯度消失或爆炸现象**

![image-20260906093519569](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906093519569.png)

![img](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typorace3256be-150f-4940-a966-bc511d1f7bcb.png)

### 2.3.2 模型内部结构

三个门：遗忘门、输入门、输出门，控制信息的进、出、留

一个记忆细胞（cell）：专门存重要的信息，相当于“长期记忆本”

变量含义：

|               变量名               |          解释含义           |                      作用                       |
| :--------------------------------: | :-------------------------: | :---------------------------------------------: |
|           $\mathbf{x}_t$           |  当前时间步 $t$ 的输入向量  |             提供当前时刻的外部信息              |
|         $\mathbf{h}_{t-1}$         | 上一时间步 $t-1$ 的隐藏状态 |           携带历史信息，参与各门计算            |
|             $C_{t-1}$              |    上一时间步的细胞状态     |                传递长期记忆信息                 |
|               $f_t$                |         遗忘门输出          |  决定从细胞状态中丢弃多少旧信息，取值 $[0,1]$   |
|               $i_t$                |         输入门输出          |        决定将多少新候选信息写入细胞状态         |
|           $\tilde{C}_t$            |        候选细胞状态         |   由当前输入和上一隐藏状态生成的新信息候选值    |
|               $C_t$                |        当前细胞状态         |     更新后的长期记忆，结合遗忘与输入门结果      |
|               $o_t$                |         输出门输出          |      决定细胞状态的哪些部分输出到隐藏状态       |
|               $h_t$                |        当前隐藏状态         |  当前时间步的输出，传递给下一时间步或用于预测   |
|           $\mathbf{W}_f$           |       遗忘门权重矩阵        |             线性变换用于计算遗忘门              |
|           $\mathbf{W}_i$           |       输入门权重矩阵        |             线性变换用于计算输入门              |
|           $\mathbf{W}_C$           |    候选细胞状态权重矩阵     |          线性变换用于生成候选细胞状态           |
|           $\mathbf{W}_o$           |       输出门权重矩阵        |             线性变换用于计算输出门              |
|           $\mathbf{b}_f$           |       遗忘门偏置向量        |                 调整遗忘门输出                  |
|           $\mathbf{b}_i$           |       输入门偏置向量        |                 调整输入门输出                  |
|           $\mathbf{b}_C$           |    候选细胞状态偏置向量     |                调整候选细胞状态                 |
|           $\mathbf{b}_o$           |       输出门偏置向量        |                 调整输出门输出                  |
|              $\sigma$              |      Sigmoid 激活函数       |      将门控值压缩到 $[0,1]$，表示保留比例       |
|              $\tanh$               |      双曲正切激活函数       | 将值压缩到 $[-1,1]$，用于生成候选状态和隐藏状态 |
|              $\odot$               | 逐元素乘法（Hadamard 乘积） |        实现门控机制，对向量元素分别加权         |
| $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ |          拼接向量           | 将上一隐藏状态与当前输入拼接，作为门计算的输入  |



LSTM 单元的计算公式如下：

#### 2.3.2.1 遗忘门

**遗忘门**：决定丢弃哪些旧信息，更新长期记忆本
$$
f_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$
$f_t$：遗忘门输出，$\mathbf{x}_t$：当前时间步 $t$ 的输入向量，$\mathbf{h}_{t-1}$：上一时间步 $t-1$ 的隐藏状态

$C_{t-1}$：上次细胞状态

![image-20260906100525225](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906100525225.png)

结论：遗忘门值 接近1 --> 保留这条旧记忆

​			遗忘门值 接近0 --> 忘掉这条旧记忆

#### 2.3.2.2 输出门

**输入门**：决定更新哪些信息，筛选新信息
$$
i_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$
$i_t$：输入门输出，$\mathbf{x}_t$：当前时间步 $t$ 的输入向量，$\mathbf{h}_{t-1}$：上一时间步 $t-1$ 的隐藏状态

![image-20260906103646602](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906103646602.png)



#### 2.3.2.3 候选细胞状态

**候选细胞状态**：创建新的候选值，即可能要存入的新信息
$$
\tilde{C}_t = \tanh(\mathbf{W}_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)
$$

#### 2.3.2.4 更新细胞状态

**更新细胞状态**：结合遗忘门和输入门，存储关键信息，能跨很多时间步传递，解决**长序列记忆问题**
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$
$C_t$：当前细胞状态，$C_{t-1}$：上次细胞状态，$\tilde{C}_t$ ：候选细胞状态 ，$f_t$：遗忘门输出，$i_t$：输入门输出

![image-20260906104545839](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906104545839.png)



#### 2.3.2.5 输出门

**输出门**：筛选要传递的信息，决定输出什么，传递给下一个管家
$$
o_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$
$o_t$：输出门输出， $\mathbf{x}_t$：当前时间步 $t$ 的输入向量，$\mathbf{h}_{t-1}$：上一时间步 $t-1$ 的隐藏状态



#### 2.3.2.6 隐藏状态

**隐藏状态**：基于更新后的细胞状态，输出的时候选择性输出，不输出的就是隐藏
$$
h_t = o_t \odot \tanh(C_t)
$$
其中 $\sigma$ 是 sigmoid 函数，输出 0 到 1 之间的值，表示信息的保留比例；$\odot$ 表示逐元素乘法；$[\mathbf{h}_{t-1}, \mathbf{x}_t]$ 表示拼接。

**结构图文字描述**：  

LSTM 单元内部包含一个细胞状态 $C_t$，它像传送带一样贯穿整个序列，信息可以很容易地流动而不易被改变。三个门分别控制：遗忘门决定从细胞状态中丢弃什么信息，输入门决定将哪些新信息存入细胞状态，输出门决定输出什么信息。

### 2.3.3 模型API

PyTorch 提供 `torch.nn.LSTM`，参数与 `nn.RNN` 类似，常见用法：

```python
import torch
import torch.nn as nn

# 定义函数, 演示LSTM模型的用法
def lstm_api():
    # 1. 创建LSTM模型对象.
    # 参1: 词向量维度(输入的维度), 参2: 隐藏层的维度(输出的维度), 参3: LSTM的层数.  参4: 是否双向.
    lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1, bidirectional=False)

    # 2. 构建输入张量.
    # 参1: 句子的长度(seq_len), 参2: 批次大小(batch_size), 参3: 词向量维度(input_size).
    input = torch.randn(4, 3, 5)

    # 3. 初始化隐藏层 和 细胞状态.
    # 参1: LSTM层数(隐藏层层数), 参2: 批次大小(batch_size), 参3: 隐藏层的维度(hidden_size).
    h0 = torch.randn(1, 3, 6)
    c0 = torch.randn(1, 3, 6)

    # 4. 模型计算.
    # 实参列表: 参1(input) -> 输入张量,  参2 -> 隐藏状态 和 细胞状态的元组形式.
    # 返回值:   output -> 本次的输出结果, (hn,cn) -> 最后1个时间步的隐藏状态和细胞状态.
    output, (hn, cn) = lstm(input, (h0, c0)) # 不写入h0,c0, 则默认为0

    # 5. 打印结果.
    print(f'output: {output}, {output.shape}')      # shape: (4, 3, 6)
    print(f'hn: {hn}, {hn.shape}')                  # shape: (1, 3, 6)
    print(f'cn: {cn}, {cn.shape}')                  # shape: (1, 3, 6)

if __name__ == '__main__':
    lstm_api()
```

### 2.3.4 优缺点

**优点**：

- **有效缓解梯度消失**：通过细胞状态和门控机制，梯度可以更稳定地传递，能够学习长期依赖，但是并不能完全解决这个问题
- **灵活的信息控制**：可以选择记住或遗忘信息，适合处理长序列。
- **广泛应用**：在语音识别、机器翻译、文本生成等任务中表现出色。

**缺点**：

- **参数较多**：每个 LSTM 单元有 4 个权重矩阵（每个门一个），计算量和内存开销较大。
- **训练速度慢**：与 RNN 类似，难以并行化，训练时间较长。
- **可能过拟合**：在数据量不足时容易过拟合，需要正则化手段。

---



## 2.4 双向BI-LSTM模型

**没有改变**LSTM模型的任何内部结构

将文本内容 **从左到右** 和 **从右到左** 做两次 LSTM处理，将最终结果张量进行**拼接**，作为最终输出

![image-20260906112020597](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed/Typoraimage-20260906112020597.png)

缺点：模型参数和计算量会增加一倍

api：在参4选择true开启

```python
# 参1: 词向量维度(输入的维度), 参2: 隐藏层的维度(输出的维度), 参3: LSTM的层数.  参4: 是否双向.
lstm = nn.LSTM(input_size=5, hidden_size=6, num_layers=1, bidirectional=False)
```



## 2.5 GRU模型

### 2.5.1 模型介绍

门控循环单元（Gated Recurrent Unit, GRU）是 LSTM 的一种变体。GRU 简化了 LSTM 的结构，将遗忘门和输入门合并为一个**更新门**，并去除了细胞状态，只保留隐藏状态。因此，GRU 参数更少，训练更快，同时在许多任务上能达到与 LSTM 相近的性能。

![image-20260906133707552](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906133707552.png)

### 2.5.2 模型内部结构

两个门：重置门和更新门-->比LSTM少一个门，更简单

一个隐藏状态：存储关键信息，比LSTM的记忆细胞更简单

变量解释如下：

|       变量名       |          解释含义           |                       作用                       |
| :----------------: | :-------------------------: | :----------------------------------------------: |
|   $\mathbf{x}_t$   |  当前时间步 $t$ 的输入向量  |              提供当前时刻的外部信息              |
| $\mathbf{h}_{t-1}$ |    上一时间步的隐藏状态     |        携带历史信息，参与门和候选状态计算        |
|       $z_t$        |         更新门输出          |      控制保留多少历史隐藏状态，取值 $[0,1]$      |
|       $r_t$        |         重置门输出          |      控制是否忽略上一隐藏状态，影响候选状态      |
|   $\tilde{h}_t$    |        候选隐藏状态         | 基于当前输入和（可能重置的）历史信息计算的新状态 |
|       $h_t$        |        当前隐藏状态         |   最终输出，结合更新门对历史状态和候选状态加权   |
|   $\mathbf{W}_z$   |       更新门权重矩阵        |              线性变换用于计算更新门              |
|   $\mathbf{W}_r$   |       重置门权重矩阵        |              线性变换用于计算重置门              |
|    $\mathbf{W}$    |    候选隐藏状态权重矩阵     |           线性变换用于生成候选隐藏状态           |
|   $\mathbf{b}_z$   |       更新门偏置向量        |                  调整更新门输出                  |
|   $\mathbf{b}_r$   |       重置门偏置向量        |                  调整重置门输出                  |
|    $\mathbf{b}$    |    候选隐藏状态偏置向量     |                 调整候选隐藏状态                 |
|      $\sigma$      |      Sigmoid 激活函数       |       将门控值压缩到 $[0,1]$，表示保留比例       |
|      $\tanh$       |      双曲正切激活函数       |           将候选状态值压缩到 $[-1,1]$            |
|      $\odot$       | 逐元素乘法（Hadamard 乘积） |         实现门控机制，对向量元素分别加权         |
|     $1 - z_t$      |       更新门的互补值        |  与 $z_t$ 配合，实现历史状态与新状态的加权平均   |



GRU 单元的计算公式如下：

#### 2.5.2.1 更新门

**更新门**：控制前一时刻的信息有多少被保留到当前状态，决定新旧记忆如何融合
$$
z_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)
$$
门值 接近1 --> 少保留旧记忆，多加新信息

门值 接近0 --> 多保留旧记忆，少加新信息

#### 2.5.2.2 重置门

**重置门**：控制前一时刻的隐藏状态如何参与候选隐藏状态的计算，决定要不要擦除旧记忆
$$
r_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)
$$
门值 接近1 --> 保留这条旧记忆

门值 接近0 --> 少保留旧记忆，多加新信息

#### 2.5.2.3 候选隐藏状态

**候选隐藏状态**：生成新记忆候选
$$
\tilde{h}_t = \tanh(\mathbf{W} [r_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b})
$$

#### 2.5.2.4 最终隐藏状态

**最终隐藏状态**：结合更新门和候选状态，新旧记忆融合
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$
其中 $\sigma$ 是 sigmoid 函数，$\odot$ 表示逐元素乘法。更新门 $z_t$ 决定保留多少历史信息，重置门 $r_t$ 决定是否忽略之前的状态。

数值 接近1 --> 少保留旧记忆，多加新信息

数值 接近0 --> 多保留旧记忆，少加新信息

**结构特点**：

- 只有两个门（更新门和重置门），比 LSTM 少一个门和细胞状态。
- 当重置门为 0 时，候选隐藏状态只依赖当前输入，忽略历史。
- 当更新门为 1 时，完全保留上一状态，不更新。

### 2.5.3 模型API

PyTorch 提供 `torch.nn.GRU`，用法与 LSTM 类似，但只需提供初始隐藏状态，无需细胞状态。

```python
import torch
import torch.nn as nn


# 定义函数, 演示: GRU代码实现
def gru_api():
    # 1. 创建GRU模型对象.
    # 参1: 输入特征维度(词向量维度), 参2: 隐藏层维度(输出维度), 参3: 层数
    gru = nn.GRU(input_size=5, hidden_size=6, num_layers=1)

    # 2. 创建输入数据.
    # 参1: 句子长度(seq_len), 参2: 批次大小(batch_size), 参3: 输入特征维度(input_size)
    input = torch.randn(2, 3, 5)

    # 3. 创建初始隐藏状态.用zeros默认全0
    # 参1: 层数(num_layers), 参2: 批次大小(batch_size), 参3: 隐藏层维度(hidden_size)
    h0 = torch.zeros(1, 3, 6)

    # 4. 运行GRU模型.
    output, hn = gru(input, h0)

    # 5. 输出结果.
    print(f'output: {output}, output.shape: {output.shape}')  # shape: (2, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')          # shape: (1, 3, 6)
    print(f'gru模型: {gru}')                                   # GRU(5, 6)


# todo 2. 测试代码
if __name__ == '__main__':
    gru_api()
```

### 2.5.4 优缺点

**优点**：

- **参数更少**：比 LSTM 少一个门，参数数量约为 LSTM 的 3/4，计算效率更高。
- **训练速度更快**：由于结构简化，收敛速度通常更快。
- **性能接近 LSTM**：在许多任务上表现与 LSTM 相当，甚至更好。
- 和LSTM一样，可以处理长序列信息，能够有效缓解梯度消失和爆炸问题

**缺点**：

- **表达能力可能略弱**：在某些复杂任务上，LSTM 可能略优于 GRU，但差异通常不大。
- **对超参数敏感**：不同任务上 GRU 和 LSTM 的表现可能有所差异，需要实验选择。
- 不能完全解决梯度消失和爆炸问题，不能并行计算



## 2.6 RNN案例——人名分类器

### 2.6.1 案例要求和数据说明

![image-20260906141858398](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906141858398.png)



### 2.6.2 基本实现分析

![image-20260906142917586](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906142917586.png)



### 2.6.3 代码实现

#### 2.6.3.1 数据处理

导入相关包，将原始数据进行one-hot编码，原始数据 -> 数据集对象TensorDataset -> 数据加载器DataLoader

代码如下：

```python
import torch                                        # 张量计算相关
import torch.nn as nn                               # 神经网络模块, 各种模型的层, 组件...
import torch.nn.functional as F                     # 常用的函数库...
import torch.optim as optim                         # 优化器模块
from  torch.utils.data import Dataset, DataLoader   # 数据集对象, 数据加载器
import string                                       # 字符串处理模块.
import time                                         # 时间模块.
import matplotlib.pyplot as plt                     # 绘图模块.
from tqdm import tqdm                               # 进度条

# 解决绘图时, 中文乱码问题.
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 获取所有的常用字符 -> 包括 字母 + 符号
all_letters = string.ascii_letters + " .,;'"        # 52个字母(大小写形式) + '空格 点 逗号 分号 单引号'

# 获取常用的字符的数量
n_letters = len(all_letters)

# 国家名 种类数.
categories = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish','Irish', 
              'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch',
                'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']

# 国家名 个数.
category_num = len(categories) # 18

# 定义函数, 读取源数据到内存
def read_data(file_path):
    """
    读取源数据到内存中, 并把 特征(人名) 和 标签(国家) 分别存储到两个列表中.
    :return: my_list_x: 存储的人名(特征), my_list_y: 存储的国家名(标签)
    """
    # 1. 创建两个列表, 分别存储: 人名(特征), 国家名(标签)
    my_list_x, my_list_y = [], []

    # 2. 关联文件, 并读取其内容(逐行读取)
    with open(file_path, 'r', encoding='utf-8') as f:
        # 3. 遍历, 获取到每一行的数据.
        for line in f.readlines():
            # 4. 过滤无效数据, 整行的长度 小于等于5, 就过滤掉.  整行长度 = 人名 + '\t' + 国家名
            if len(line) <= 5:
                continue
            # 5. 添加到对应的列表中.
            x, y = line.strip().split('\t')
            # 6. 添加到对应的列表中.
            my_list_x.append(x)
            my_list_y.append(y)

    # 7. 返回解析后的 样本 和 标签.
    return my_list_x, my_list_y


# 创建数据集对象, 原始数据 -> 数据集对象TensorDataset -> 数据加载器DataLoader
class NameClassDataset(Dataset):
    # 1. 初始化函数, 接收: 样本和标签数据, 初始化数据集基本属性.
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x          # 存储样本数据列表
        self.my_list_y = my_list_y          # 存储标签数据列表
        self.sample_len = len(my_list_x)    # 计算样本总数并存储, 20074

    # 2. 定义函数, 用于获取样本总数
    def __len__(self):
        return self.sample_len

    # 3. 定义函数, 实现根据指定索引, 获取其对应的样本.
    def __getitem__(self, index):
        """
        根据指定的索引, 获取其对应的样本, 并进行 one-hot编码 和 张量转换.
        :param index: 样本索引
        :return:  tensor_x: 人名(特征)的one-hot编码,  tensor_y: 国家(标签)的张量表示
        """
        # 1. 索引边界校验, 确保索引在合法范围.    [0, self.sample_len - 1]
        index = min(max(index, 0), self.sample_len - 1)

        # 2. 按照索引获取原始样本 和 标签.
        x = self.my_list_x[index]       # 例如: Ding      ->  (4, 57) 每个字母都要转成57个one-hot
        y = self.my_list_y[index]       # 例如: Chinese   ->  18个国家中的某个索引, 例如: 6

        # 3. 人名数据转换为 one-hot编码.
        # 3.1 生成全0张量
        tensor_x = torch.zeros(len(x), n_letters)       # 例如: [4, 57]

        # 3.2 遍历人名, 获取每个字母, 生成one-hot张量.
        # enumerate(x):一边拿字母的索引(在当前单词的位置)，一边拿字母
        for li, letter in enumerate(x):
            # 3.2.1 获取字母在 全局字母表中的索引位置
            letter_index = all_letters.find(letter)
            # 3.2.2 在对应位置设置为1 -> 即: one-hot编码
            tensor_x[li][letter_index] = 1

        # 4. 国家数据转换为 张量.
        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)

        # 5. 返回结果
        return tensor_x, tensor_y

# 定义函数, 获取数据加载器对象 思路: Tensor -> TensorDataset -> DataLoader
def get_dataloader():
    # 1. 读取数据文件, 获取: 样本(人名)列表 和 标签(国家名)列表.
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 2. 创建数据集对象
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 3. 创建数据加载器对象, 用于批量加载和处理数据.
    # 参1: 数据集对象(Dataset),  参2: 批次大小(每批多少条数据), 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    my_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

    # 4. 测试数据加载器, 打印第一批数据(某一个样本的) 形状 和 内容
    for x, y in my_dataloader:
        print(f'x.shape: {x.shape}, x: {x}')        # 人名的张量形状和内容.
        print(f'y.shape: {y.shape}, y: {y}')        # 国家张量形状和内容.
        break       # 仅打印第1批次数据, 用于查看, 避免全部输出.

    # return my_dataloader


# todo n. 测试代码
if __name__ == '__main__':
    # 1. 读取数据
    # my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 2. 测试: 数据加载器.
    get_dataloader()
```



#### 2.6.3.2 LogSoftmax介绍

对softmax的结果再取对数

可以解决下溢问题，让交叉熵的计算简化。

示例代码如下：

```python
import torch
import torch.nn as nn

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 1. 创建数据 -> 模拟模型输出的原始分时(logits), 表示3个分类的预测值, 即: 全连接层的处理后的结果.
output = torch.tensor([3.2, 5.1, -1.7])

# 2. 用LogSoftmax()函数计算
# 2.1 创建LogSoftmax()函数对象 dim=0 按照列操作
log_softmax = nn.LogSoftmax(dim=0)

# 2.2 具体的计算, 先softmax(), 然后log()
log_probs = log_softmax(output)

# 2.3 打印结果
print(f'计算结果(对数概率): {log_probs}')       # tensor([-2.0404, -0.1404, -6.9404])


# 3. 手动验算, 先softmax(), 然后log()
# 3.1 创建softmax()函数对象
softmax = torch.softmax(output, dim=0)
print(f'softmax()计算结果: {softmax}')        # tensor([0.1300, 0.8690, 0.0010])

# 3.2 手动计算log() -> 对softmax()结果, 取自然对数, 得到: 对数概率.
log_softmax_probs = torch.log(softmax)
print(f'计算结果(对数概率): {log_softmax_probs}')   # tensor([-2.0404, -0.1404, -6.9404])

```



#### 2.6.3.3 RNN实现



```python
# RNN模型训练.
def train_rnn():
    # 1. 数据准备动作.
    # 1.1 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象.
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 2. 模型与优化器初始化.
    # 2.1 定义模型参数,
    # 参1: 输入维度(字符表大小), 参2: 隐藏层维度, 参3: 输出维度(国家数量)
    input_size, n_hidden, output_size = n_letters, 128, category_num       # 等价于: 57, 128, 18

    # 2.2 创建模型对象.
    my_rnn = My_RNN(input_size, n_hidden, output_size)

    # 2.3 定义损失函数和优化器.
    criterion = nn.NLLLoss()    # 如果你用了CrossEntropyLoss(), 则它 = NLLLoss() + LogSoftmax()
    optimizer = optim.Adam(my_rnn.parameters(), lr=my_lr)

    # 3. 训练过程 -> 参数初始化
    start_time = time.time()        # 模型开始训练时间
    total_iter_num = 0              # 已训练的样本数
    total_loss = 0.0                # 已训练的损失和
    total_loss_list = []            # 每100个样本求一次平均损失, 形成: 损失列表
    total_acc_num = 0               # 已训练的样本, 预测准确总数
    total_acc_list = []             # 每100个样本求一次平均准确率, 形成: 准确率列表

    # 4. 具体的训练过程, 按轮数遍历数据集.
    for epoch in range(epochs):     # epoch: 第几轮

        print(f'\n开始第{epoch + 1}/{epochs} 轮训练...')

        # 4.1 创建数据集加载器对象, 随机打乱数据集.
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

        # 4.2 样本迭代训练,  即: 本轮具体的每批次训练
        for i, (x, y) in enumerate(tqdm(train_dataloader)):     # 优化点3: 这里加入进度条.
            # 4.3 前向传播, 计算结果.
            output, hidden = my_rnn(x[0], my_rnn.init_hidden())
            # 4.4 计算损失.
            my_loss = criterion(output, y)
            # 4.5 三剑客 -> 梯度清零, 反向传播, 优化器更新参数.
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 4.6 统计训练结果(指标统计)
            total_iter_num += 1             # 训练的样本数 + 1
            total_loss += my_loss.item()    # 累计损失值

            # 4.7 计算当前样本预测准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y else 0)        # 统计: 预测正确的样本数

            # 4.8 统计: 每100个样本求一次平均损失, 准确率 形成: 损失列表, 准确率列表.
            if total_iter_num % 100 == 0:
                # 走这里, 说明100步了, 计算: 平均损失.
                avg_loss = total_loss / total_iter_num      # 总损失 / 总样本数
                # 把上述的平均损失, 添加到: 损失列表.
                total_loss_list.append(avg_loss)

                # 计算准确率, 即: 预测正确的 / 总样本数, 并添加到: 准确率列表.
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 4.9 每2000步(个样本), 打印训练日志.
            if total_iter_num % 2000 == 0:
                # 计算平均损失.
                avg_loss = total_loss / total_iter_num
                # 计算模型训练耗时
                end_time = int(time.time() - start_time)
                # 输出训练日志.
                print(f'轮次: {epoch + 1}, 训练的样本数: {total_iter_num}, 平均损失: {avg_loss:.4f}, 耗时: {end_time}s, 准确率: {avg_acc:.4f}')

        # 4.10 走到这里, 说明一轮训练完毕 -> 保存模型.
        torch.save(my_rnn.state_dict(), f'./model/my_rnn_GNC_{epoch + 1}.bin')

    # 5. 走到这里, 训练结束, 返回统计结果.
    total_time = int(time.time() - start_time)
    print(f'训练完成, 总耗时: {total_time}s, 总训练了 {total_iter_num}个样本!!')

    # 6. 优化4: 你可以把下述返回的三个值(损失列表, 训练总耗时, 准确率列表), 存储到文件中.
    #          因为一会儿我们会 可视化3个模型的训练结果, 如果没有存储的话, 会把 训练动作从新跑一次.

    # 7. 返回结果: 损失列表, 训练总耗时, 准确率列表.
    return total_loss_list, total_time, total_acc_list

```



#### 2.6.3.4 LSTM实现

```python
# LSTM模型训练.
def train_lstm():
    # 1. 数据准备动作.
    # 1.1 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象.
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 2. 模型与优化器初始化.
    # 2.1 定义模型参数,
    # 参1: 输入维度(字符表大小), 参2: 隐藏层维度, 参3: 输出维度(国家数量)
    input_size, n_hidden, output_size = n_letters, 128, category_num       # 等价于: 57, 128, 18

    # 2.2 创建模型对象.
    my_rnn = My_LSTM(input_size, n_hidden, output_size)

    # 2.3 定义损失函数和优化器.
    criterion = nn.NLLLoss()    # 如果你用了CrossEntropyLoss(), 则它 = NLLLoss() + LogSoftmax()
    optimizer = optim.Adam(my_rnn.parameters(), lr=my_lr)

    # 3. 训练过程 -> 参数初始化
    start_time = time.time()        # 模型开始训练时间.
    total_iter_num = 0              # 已训练的样本数.
    total_loss = 0.0                # 已训练的损失和
    total_loss_list = []            # 每100个样本求一次平均损失, 形成: 损失列表.
    total_acc_num = 0               # 已训练的样本, 预测准确总数
    total_acc_list = []             # 每100个样本求一次平均准确率, 形成: 准确率列表.

    # 4. 具体的训练过程, 按轮数遍历数据集.
    for epoch in range(epochs):     # epoch: 第几轮
        print(f'\n开始第{epoch + 1}/{epochs} 轮训练...')
        # 4.1 创建数据集加载器对象, 随机打乱数据集.
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
        # 4.2 样本迭代训练,  即: 本轮具体的每批次训练
        for i, (x, y) in enumerate(tqdm(train_dataloader)):     # 优化点3: 这里加入进度条.
            # 4.3 前向传播, 计算结果.
            hidden, c = my_rnn.init_hidden()
            output, hidden, c = my_rnn(x[0], hidden, c)
            # 4.4 计算损失.
            my_loss = criterion(output, y)
            # 4.5 三剑客 -> 梯度清零, 反向传播, 优化器更新参数.
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 4.6 统计训练结果(指标统计)
            total_iter_num += 1             # 训训练的样本数 + 1
            total_loss += my_loss.item()    # 累计损失值

            # 4.7 计算当前样本预测准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y else 0)        # 统计: 预测正确的样本数

            # 4.8 统计: 每100个样本求一次平均损失, 准确率 形成: 损失列表, 准确率列表.
            if total_iter_num % 100 == 0:
                # 走这里, 说明100步了, 计算: 平均损失.
                avg_loss = total_loss / total_iter_num      # 总损失 / 总样本数
                # 把上述的平均损失, 添加到: 损失列表.
                total_loss_list.append(avg_loss)

                # 计算准确率, 即: 预测正确的 / 总样本数, 并添加到: 准确率列表.
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 4.9 每2000步(个样本), 打印训练日志.
            if total_iter_num % 2000 == 0:
                # 计算平均损失.
                avg_loss = total_loss / total_iter_num
                # 计算模型训练耗时
                end_time = int(time.time() - start_time)
                # 输出训练日志.
                print(f'轮次: {epoch + 1}, 训练的样本数: {total_iter_num}, 平均损失: {avg_loss:.4f}, 耗时: {end_time}s, 准确率: {avg_acc:.4f}')

        # 4.10 走到这里, 说明一轮训练完毕 -> 保存模型.
        torch.save(my_rnn.state_dict(), f'./model/my_lstm_GNC_{epoch + 1}.bin')

    # 5. 走到这里, 训练结束, 返回统计结果.
    total_time = int(time.time() - start_time)
    print(f'训练完成, 总耗时: {total_time}s, 总训练了 {total_iter_num}个样本!!')

    # 6. 优化4: 你可以把下述返回的三个值(损失列表, 训练总耗时, 准确率列表), 存储到文件中.
    #          因为一会儿我们会 可视化3个模型的训练结果, 如果没有存储的话, 会把 训练动作从新跑一次.

    # 7. 返回结果: 损失列表, 训练总耗时, 准确率列表.
    return total_loss_list, total_time, total_acc_list

```



#### 2.6.3.5 GRU实现

```python
# GRU模型训练.
def train_gru():
    # 1. 数据准备动作.
    # 1.1 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象.
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 2. 模型与优化器初始化.
    # 2.1 定义模型参数,
    # 参1: 输入维度(字符表大小), 参2: 隐藏层维度, 参3: 输出维度(国家数量)
    input_size, n_hidden, output_size = n_letters, 128, category_num       # 等价于: 57, 128, 18

    # 2.2 创建模型对象.
    my_rnn = My_GRU(input_size, n_hidden, output_size)

    # 2.3 定义损失函数和优化器.
    criterion = nn.NLLLoss()    # 如果你用了CrossEntropyLoss(), 则它 = NLLLoss() + LogSoftmax()
    optimizer = optim.Adam(my_rnn.parameters(), lr=my_lr)

    # 3. 训练过程 -> 参数初始化
    start_time = time.time()        # 模型开始训练时间.
    total_iter_num = 0              # 已训练的样本数.
    total_loss = 0.0                # 已训练的损失和
    total_loss_list = []            # 每100个样本求一次平均损失, 形成: 损失列表.
    total_acc_num = 0               # 已训练的样本, 预测准确总数
    total_acc_list = []             # 每100个样本求一次平均准确率, 形成: 准确率列表.

    # 4. 具体的训练过程, 按轮数遍历数据集.
    for epoch in range(epochs):     # epoch: 第几轮
        print(f'\n开始第{epoch + 1}/{epochs} 轮训练...')
        # 4.1 创建数据集加载器对象, 随机打乱数据集.
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
        # 4.2 样本迭代训练,  即: 本轮具体的每批次训练
        for i, (x, y) in enumerate(tqdm(train_dataloader)):     # 优化点3: 这里加入进度条.
            # 4.3 前向传播, 计算结果.
            output, hidden = my_rnn(x[0], my_rnn.init_hidden())
            # 4.4 计算损失.
            my_loss = criterion(output, y)
            # 4.5 三剑客 -> 梯度清零, 反向传播, 优化器更新参数.
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 4.6 统计训练结果(指标统计)
            total_iter_num += 1             # 训训练的样本数 + 1
            total_loss += my_loss.item()    # 累计损失值

            # 4.7 计算当前样本预测准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y else 0)        # 统计: 预测正确的样本数

            # 4.8 统计: 每100个样本求一次平均损失, 准确率 形成: 损失列表, 准确率列表.
            if total_iter_num % 100 == 0:
                # 走这里, 说明100步了, 计算: 平均损失.
                avg_loss = total_loss / total_iter_num      # 总损失 / 总样本数
                # 把上述的平均损失, 添加到: 损失列表.
                total_loss_list.append(avg_loss)

                # 计算准确率, 即: 预测正确的 / 总样本数, 并添加到: 准确率列表.
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 4.9 每2000步(个样本), 打印训练日志.
            if total_iter_num % 2000 == 0:
                # 计算平均损失.
                avg_loss = total_loss / total_iter_num
                # 计算模型训练耗时
                end_time = int(time.time() - start_time)
                # 输出训练日志.
                print(f'轮次: {epoch + 1}, 训练的样本数: {total_iter_num}, 平均损失: {avg_loss:.4f}, 耗时: {end_time}s, 准确率: {avg_acc:.4f}')

        # 4.10 走到这里, 说明一轮训练完毕 -> 保存模型.
        torch.save(my_rnn.state_dict(), f'./model/my_gru_GNC_{epoch + 1}.bin')

    # 5. 走到这里, 训练结束, 返回统计结果.
    total_time = int(time.time() - start_time)
    print(f'训练完成, 总耗时: {total_time}s, 总训练了 {total_iter_num}个样本!!')

    # 6. 优化4: 你可以把下述返回的三个值(损失列表, 训练总耗时, 准确率列表), 存储到文件中.
    #          因为一会儿我们会 可视化3个模型的训练结果, 如果没有存储的话, 会把 训练动作从新跑一次.


    # 7. 返回结果: 损失列表, 训练总耗时, 准确率列表.
    return total_loss_list, total_time, total_acc_list

```



#### 2.6.3.6 三种模型可视化对比

```python
# 模型训练绘图 -> 这个函数的可视化代码你可以不写, 但是模型训练等代码要写出来, 出3张图.
def test_train_rnn_lstm_gru():
    # 1. 训练3种模型, 并获取性能指标.
    # 参1: 损失列表, 参2: 训练总耗时, 参3: 准确率列表.
    total_loss_list_rnn, total_time_rnn, total_acc_list_rnn = train_rnn()
    total_loss_list_lstm, total_time_lstm, total_acc_list_lstm = train_lstm()
    total_loss_list_gru, total_time_gru, total_acc_list_gru = train_gru()

    # 2. 绘制 损失对比曲线(评估: 模型收敛速度)
    # 2.1 创建画布.     0: 图1
    plt.figure(0, figsize=(10, 5))
    # 2.2 绘制各模型损失曲线.
    plt.plot(total_loss_list_rnn, label='RNN')
    plt.plot(total_loss_list_lstm, label='LSTM')
    plt.plot(total_loss_list_gru, label='GRU')
    # 2.3 设置图表属性
    plt.title('模型损失对比曲线')
    plt.xlabel('训练步数(每100步)')
    plt.ylabel('平均损失值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_loss_time.png')
    plt.show()

    # 3. 绘制 训练耗时对比柱状图(评估: 模型计算效率)
    # 3.1 创建画布,     1: 图2
    plt.figure(1, figsize=(10, 5))
    # 3.2 准备x轴 和 y轴标签内容.
    x_data = ['RNN', 'LSTM', 'GRU']
    y_data = [total_time_rnn, total_time_lstm, total_time_gru]

    # 3.3 绘制柱状图
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)

    # 3.4 设置图表属性
    plt.title('模型耗时对比柱状图')
    plt.savefig('./img/RNN_LSTM_GRU_time.png')
    plt.show()


    # 4. 绘制 训练准确率对比曲线(评估: 模型效果)
    # 4.1 创建画布.     2: 图3
    plt.figure(2, figsize=(10, 5))
    # 4.2 绘制各模型准确率曲线.
    plt.plot(total_acc_list_rnn, label='RNN', color='red')
    plt.plot(total_acc_list_lstm, label='LSTM', color='green')
    plt.plot(total_acc_list_gru, label='GRU', color='orange')
    # 4.3 绘制图表属性.
    plt.title('模型准确率对比曲线')
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_acc.png')
    plt.show()
```

结果图如下：

![image-20260906195033035](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906195033035.png)

![image-20260906195059248](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906195059248.png)

![image-20260906195108606](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260906195108606.png)



#### 2.6.3.7 模型预测

```python
# todo 10. 模型预测.
# todo 10.1 RNN模型预测
# todo 10.1.1. 定义遍历, 记录模型的参数的路径.
my_rnn_path = './model/my_rnn_GNC_1.bin'
my_lstm_path = './model/my_lstm_GNC_1.bin'
my_gru_path = './model/my_gru_GNC_1.bin'

# todo 10.1.2 定义函数, 将要预测的人名 转成 one-hot编码, 例如: 'zhang' -> [5, 57]
def lineToTensor(line):
    # 1. 初始化张量, [文本长度, 字符表长度]
    tensor_x = torch.zeros(len(line), n_letters)

    # 2. 遍历文本, 获取到每个字符及其索引.
    for i, letter in enumerate(line):
        # 3. 查看字符在全局字母表中的位置(索引)
        letter_index = all_letters.find(letter)
        # 4. 在张量的对应位置改为1, 完成: one-hot编码
        tensor_x[i][letter_index] = 1

    # 5. 返回结果.
    return tensor_x         # 即: 'zhang' -> [5, 57]


# todo 10.1.3 定义函数, 实现: RNN预测.
def predict_rnn(x):
    # 1. 定义遍历, 记录模型相关参数.
    n_letters, n_hidden, n_categories = 57, 128, 18
    # 2. 把输入的文字转成 one-hot编码.
    x_tensor = lineToTensor(x)
    # 3. 创建模型对象.
    my_rnn = My_RNN(n_letters, n_hidden, n_categories)
    # 4. 加载模型参数.
    my_rnn.load_state_dict(torch.load(my_rnn_path))
    # 5. 进行预测, 不计算梯度.  -> 节省内存和计算机资源.
    with torch.no_grad():
        # 5.1 模型预测
        output, hidden = my_rnn(x_tensor, my_rnn.init_hidden())
        # 5.2 从预测结果中, 获取前3个最大的元素.
        # 参1(k):   取前3个最大的元素.
        # 参2(dim): 获取概率最大的元素所在的维度.
        # 参3(largest): 获取概率最大的元素.
        topv, topi = output.topk(3, 1, True)
        # 5.3 打印待预测文本.
        print(f'rnn(待预测文本): {x}')

        # 5.4 解析预测结果.
        for i in range(3):
            value = topv[0][i].item()           # 概率值 -> Python的标量
            category_idx  = topi[0][i].item()   # 类别索引
            category = categories[category_idx] # 类别名称.
            print(f'value: {value}, category: {category}')
```



#### 2.6.3.8 完整代码

```python
import torch                                        # 张量计算相关
import torch.nn as nn                               # 神经网络模块, 各种模型的层, 组件...
import torch.nn.functional as F                     # 常用的函数库...
import torch.optim as optim                         # 优化器模块
from  torch.utils.data import Dataset, DataLoader   # 数据集对象, 数据加载器
import string                                       # 字符串处理模块.
import time                                         # 时间模块.
import matplotlib.pyplot as plt                     # 绘图模块.
from tqdm import tqdm                               # 进度条

# 解决绘图时, 中文乱码问题.
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 获取所有的常用字符 -> 包括 字母 + 符号
all_letters = string.ascii_letters + " .,;'"        # 52个字母(大小写形式) + '空格 点 逗号 分号 单引号'

# 获取常用的字符的数量
n_letters = len(all_letters)

# 国家名 种类数.
categories = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish','Irish', 
              'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch',
                'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']

# 国家名 个数.
category_num = len(categories) # 18

# 定义函数, 读取源数据到内存
def read_data(file_path):
    """
    读取源数据到内存中, 并把 特征(人名) 和 标签(国家) 分别存储到两个列表中.
    :return: my_list_x: 存储的人名(特征), my_list_y: 存储的国家名(标签)
    """
    # 1. 创建两个列表, 分别存储: 人名(特征), 国家名(标签)
    my_list_x, my_list_y = [], []

    # 2. 关联文件, 并读取其内容(逐行读取)
    with open(file_path, 'r', encoding='utf-8') as f:
        # 3. 遍历, 获取到每一行的数据.
        for line in f.readlines():
            # 4. 过滤无效数据, 整行的长度 小于等于5, 就过滤掉.  整行长度 = 人名 + '\t' + 国家名
            if len(line) <= 5:
                continue
            # 5. 添加到对应的列表中.
            x, y = line.strip().split('\t')
            # 6. 添加到对应的列表中.
            my_list_x.append(x)
            my_list_y.append(y)

    # 7. 返回解析后的 样本 和 标签.
    return my_list_x, my_list_y

# 创建数据集对象, 原始数据 -> 数据集对象TensorDataset -> 数据加载器DataLoader
class NameClassDataset(Dataset):
    # 1. 初始化函数, 接收: 样本和标签数据, 初始化数据集基本属性.
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x          # 存储样本数据列表
        self.my_list_y = my_list_y          # 存储标签数据列表
        self.sample_len = len(my_list_x)    # 计算样本总数并存储, 20074

    # 2. 定义函数, 用于获取样本总数
    def __len__(self):
        return self.sample_len

    # 3. 定义函数, 实现根据指定索引, 获取其对应的样本.
    def __getitem__(self, index):
        """
        根据指定的索引, 获取其对应的样本, 并进行 one-hot编码 和 张量转换.
        :param index: 样本索引
        :return:  tensor_x: 人名(特征)的one-hot编码,  tensor_y: 国家(标签)的张量表示
        """
        # 1. 索引边界校验, 确保索引在合法范围.    [0, self.sample_len - 1]
        index = min(max(index, 0), self.sample_len - 1)

        # 2. 按照索引获取原始样本 和 标签.
        x = self.my_list_x[index]       # 例如: Ding      ->  (4, 57) 每个字母都要转成57个one-hot
        y = self.my_list_y[index]       # 例如: Chinese   ->  18个国家中的某个索引, 例如: 6

        # 3. 人名数据转换为 one-hot编码.
        # 3.1 生成全0张量
        tensor_x = torch.zeros(len(x), n_letters)       # 例如: [4, 57]

        # 3.2 遍历人名, 获取每个字母, 生成one-hot张量.
        # enumerate(x):一边拿字母的索引(在当前单词的位置)，一边拿字母
        for li, letter in enumerate(x):
            # 3.2.1 获取字母在 全局字母表中的索引位置
            letter_index = all_letters.find(letter)
            # 3.2.2 在对应位置设置为1 -> 即: one-hot编码
            tensor_x[li][letter_index] = 1

        # 4. 国家数据转换为 张量.
        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)

        # 5. 返回结果
        return tensor_x, tensor_y

# 定义函数, 获取数据加载器对象 思路: Tensor -> TensorDataset -> DataLoader
def get_dataloader():
    # 1. 读取数据文件, 获取: 样本(人名)列表 和 标签(国家名)列表.
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 2. 创建数据集对象
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 3. 创建数据加载器对象, 用于批量加载和处理数据.
    # 参1: 数据集对象(Dataset),  参2: 批次大小(每批多少条数据), 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    my_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

    # 4. 测试数据加载器, 打印第一批数据(某一个样本的) 形状 和 内容
    for x, y in my_dataloader:
        print(f'x.shape: {x.shape}, x: {x}')        # 人名的张量形状和内容.
        print(f'y.shape: {y.shape}, y: {y}')        # 国家张量形状和内容.
        break       # 仅打印第1批次数据, 用于查看, 避免全部输出.

    # return my_dataloader


# 搭建RNN模型
class My_RNN(nn.Module):
    # 1. 初始化函数: 输入特征维度, 隐藏层维度, 输出维度, 层数.
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 输入特征维度(对应字母表大小, 即: 57个字符)
        self.input_size = input_size
        # 1.3 隐藏层维度, 决定模型的表示能力.
        self.hidden_size = hidden_size
        # 1.4 输出维度(对应国家名数量, 即: 18个国家名)
        self.output_size = output_size
        # 1.5 层数, 默认为1.
        self.n_layers = n_layers

        # 1.6 定义RNN层, 接收输入特征 和 输出隐藏状态.
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers)

        # 1.7 定义全连接层, 将RNN的隐藏状态转换成输出.
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 1.8 定义激活函数, 将输出类别 -> 所属类别的概率分布.
        # 大白话解释: 多分类交叉熵损失函数CrossEntropyLoss(新版写法) = NLLLoss损失函数 + LogSoftmax(dim=-1)  旧版写法
        self.softmax = nn.LogSoftmax(dim=-1)        # 优化2: 如果用CrossEntropyLoss损失函数, 这行代码可以省略不写.
      
    # 2. 前向传播函数.
    # 参1: input输入张量, 当前形状为: [seq_len(4), input_size(57)] 需要转成 [seq_len, batch_size, input_size]
    # 参2: hidden(隐藏状态), 形状为: [n_layers, batch_size, hidden_size]
    def forward(self, input, hidden):
        # 2.1 调整输入张量, 添加: batch_size
        input = input.unsqueeze(1)

        # 2.2 通过RNN计算.
        # output: 所有时间步的隐藏状态   hidden: 最后1个时间步的隐藏状态
        output, hn = self.rnn(input, hidden)

        # 2.3 提取最后1个时间步的隐藏状态.
        tmp_output = output[-1]             # 形状为: [batch_size, hidden_size]

        # 2.4 通过全连接层, 获取输出.
        tmp_output = self.linear(tmp_output)

        # 2.5 数据通过激活函数, 映射到概率分布, 并返回.
        return self.softmax(tmp_output), hn

    # 3. 初始化隐藏状态, 创建全0的初始化隐藏状态.
    def init_hidden(self):
        # 参1: 隐藏层层数, 参2: 批次大小, 参3: 隐藏层维度.
        return torch.zeros(self.n_layers, 1, self.hidden_size)

# 测试RNN模型——了解
def dm_test_myrnn():
    # 1. 实例化RNN对象
    my_rnn = My_RNN(57, 128, 18)  

    # 2. 准备测试数据, 创建1个随机张量, 模拟输入, 形状为: [seq_len人名长度, input_size词向量维度]
    input = torch.randn(6, 57)      # 测试: ouyang 欧阳
    print(f'input(输入的张量维度): {input.shape}')     # torch.Size([6, 57])

    # 3. 初始化隐藏状态
    # h0 = torch.zeros(1, 1, 128)
    h0 = my_rnn.init_hidden()       # 效果同上.

    # 4. 测试一次性输入完整的一个样本(序列数据)
    output, hn = my_rnn(input, h0)

    # 5. 打印结果.
    print(f'输出的形状: {output.shape}, 输出的内容: {output}')         # [1, 18]
    print(f'隐藏状态的形状: {hn.shape}, 隐藏状态的内容: {hn}')          #  [1, 1, 128]

# 搭建LSTM模型
class My_LSTM(nn.Module):
    # 1. 初始化函数: 输入特征维度, 隐藏层维度, 输出维度, 层数.
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 输入特征维度(对应字母表大小, 即: 57个字符)
        self.input_size = input_size
        # 1.3 隐藏层维度, 决定模型的表示能力.
        self.hidden_size = hidden_size
        # 1.4 输出维度(对应国家名数量, 即: 18个国家名)
        self.output_size = output_size
        # 1.5 层数, 默认为1.
        self.n_layers = n_layers

        # 1.6 定义LSTM层, 接收输入特征 和 输出隐藏状态.
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)

        # 1.7 定义全连接层, 将RNN的隐藏状态转换成输出.
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 1.8 定义激活函数, 将输出类别 -> 所属类别的概率分布.
        # 大白话解释: 多分类交叉熵损失函数CrossEntropyLoss(新版写法) = NLLLoss损失函数 + LogSoftmax(dim=-1)  旧版写法
        self.softmax = nn.LogSoftmax(dim=-1)        # 优化2: 如果用CrossEntropyLoss损失函数, 这行代码可以省略不写.
      
    # 2. 前向传播函数.
    # 参1: input输入张量, 当前形状为: [seq_len(4), input_size(57)] 需要转成 [seq_len, batch_size, input_size]
    # 参2: hidden(隐藏状态), 形状为: [n_layers, batch_size, hidden_size]
    def forward(self, input, hidden, c):
        # 2.1 调整输入张量, 添加: batch_size
        input = input.unsqueeze(1)

        # 2.2 通过LSTM计算.
        # output: 所有时间步的隐藏状态   hidden: 最后1个时间步的隐藏状态
        output, (hn, cn) = self.rnn(input, (hidden, c))

        # 2.3 提取最后1个时间步的隐藏状态.
        tmp_output = output[-1]             # 形状为: [batch_size, hidden_size]

        # 2.4 通过全连接层, 获取输出.
        tmp_output = self.linear(tmp_output)

        # 2.5 数据通过激活函数, 映射到概率分布, 并返回.
        return self.softmax(tmp_output), hn, cn

    # 3. 初始化隐藏状态, 创建全0的初始化隐藏状态.
    def init_hidden(self):
        # 参1: 隐藏层层数, 参2: 批次大小, 参3: 隐藏层维度.
        hidden = c = torch.zeros(self.n_layers, 1, self.hidden_size)
        return hidden, c

# 搭建GRU模型
class My_GRU(nn.Module):
    # 1. 初始化函数: 输入特征维度, 隐藏层维度, 输出维度, 层数.
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 输入特征维度(对应字母表大小, 即: 57个字符)
        self.input_size = input_size
        # 1.3 隐藏层维度, 决定模型的表示能力.
        self.hidden_size = hidden_size
        # 1.4 输出维度(对应国家名数量, 即: 18个国家名)
        self.output_size = output_size
        # 1.5 层数, 默认为1.
        self.n_layers = n_layers

        # 1.6 定义GRU层, 接收输入特征 和 输出隐藏状态.
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.n_layers)

        # 1.7 定义全连接层, 将RNN的隐藏状态转换成输出.
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 1.8 定义激活函数, 将输出类别 -> 所属类别的概率分布.
        # 大白话解释: 多分类交叉熵损失函数CrossEntropyLoss(新版写法) = NLLLoss损失函数 + LogSoftmax(dim=-1)  旧版写法
        self.softmax = nn.LogSoftmax(dim=-1)        # 优化2: 如果用CrossEntropyLoss损失函数, 这行代码可以省略不写.
      
    # 2. 前向传播函数.
    # 参1: input输入张量, 当前形状为: [seq_len(4), input_size(57)] 需要转成 [seq_len, batch_size, input_size]
    # 参2: hidden(隐藏状态), 形状为: [n_layers, batch_size, hidden_size]
    def forward(self, input, hidden):
        # 2.1 调整输入张量, 添加: batch_size
        input = input.unsqueeze(1)

        # 2.2 通过RNN计算.
        # output: 所有时间步的隐藏状态   hidden: 最后1个时间步的隐藏状态
        output, hn = self.rnn(input, hidden)

        # 2.3 提取最后1个时间步的隐藏状态.
        tmp_output = output[-1]             # 形状为: [batch_size, hidden_size]

        # 2.4 通过全连接层, 获取输出.
        tmp_output = self.linear(tmp_output)

        # 2.5 数据通过激活函数, 映射到概率分布, 并返回.
        return self.softmax(tmp_output), hn

    # 3. 初始化隐藏状态, 创建全0的初始化隐藏状态.
    def init_hidden(self):
        # 参1: 隐藏层层数, 参2: 批次大小, 参3: 隐藏层维度.
        return torch.zeros(self.n_layers, 1, self.hidden_size)

# 测试RNN, LSTM, GRU网络模型——了解
def test_rnn_lstm_gru():
    # 1. 定义遍历, 记录: 输入维度(词向量维度: 57), 隐藏层维度(128), 输出维度(18, 国家数量)
    input_size, n_hidden, output_size = n_letters, 128, category_num

    # 2. 加载数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 3. 创建数据集对象
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 4. 创建数据加载器.
    my_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
    
    # 5. 模型初始化.
    my_rnn = My_RNN(input_size, n_hidden, output_size)
    my_lstm = My_LSTM(input_size, n_hidden, output_size)
    my_gru = My_GRU(input_size, n_hidden, output_size)

    # 6. 模型结构可视化.
    print(f'RNN模型结构: {my_rnn}') 
    print(f'LSTM模型结构: {my_lstm}')
    print(f'GRU模型结构: {my_gru}')

    # 7. 测试上述的3个模型
    # 7.1 测试RNN模型
    for i, (x, y) in enumerate(my_dataloader):
        print(f'i: {i}')                        #  编号, 第i条数据
        print(f'x: {x}, x.shape: {x.shape}')    # 输入数据的词向量形式, 例如: x.shape: torch.Size([1, 10, 57])
        print(f'y: {y}, y.shape: {y.shape}')    # 输出数据(国家的编号), 例如: y: tensor([15]), y.shape: torch.Size([1])

        # 7.2 初始化隐藏状态
        hidden = my_rnn.init_hidden()           # 形状: [1, 1, 128]

        # 7.3 前向传播
        output, hidden = my_rnn(x[0], hidden)   # x[0] 等价于: [10, 57]
        print(f'RNN输出形状: {output.shape}, 预测结果: {output}')

        # 扩展: 只训练1个样本, 不然太多了, 这里看看即可.
        if i == 0:
            break

    # 7.2 测试LSTM模型
    for i, (x, y) in enumerate(my_dataloader):
        # print(f'i: {i}')  # 编号, 第i条数据
        # print(f'x: {x}, x.shape: {x.shape}')  # 输入数据的词向量形式, 例如: x.shape: torch.Size([1, 10, 57])
        # print(f'y: {y}, y.shape: {y.shape}')  # 输出数据(国家的编号), 例如: y: tensor([15]), y.shape: torch.Size([1])

        # 7.2 初始化隐藏状态.
        hidden, c = my_lstm.init_hidden()  # 形状: [1, 1, 128]

        # 7.3 前向传播.
        output, hidden, c = my_lstm(x[0], hidden, c)  # x[0] 等价于: [10, 57]
        print(f'LSTM输出形状: {output.shape}, 预测结果: {output}')

        # 扩展: 只训练1个样本, 不然太多了, 这里看看即可.
        if i == 0:
            break

    # 7.3 测试GRU模型
    for i, (x, y) in enumerate(my_dataloader):
        # print(f'i: {i}')  # 编号, 第i条数据
        # print(f'x: {x}, x.shape: {x.shape}')  # 输入数据的词向量形式, 例如: x.shape: torch.Size([1, 10, 57])
        # print(f'y: {y}, y.shape: {y.shape}')  # 输出数据(国家的编号), 例如: y: tensor([15]), y.shape: torch.Size([1])

        # 7.2 初始化隐藏状态.
        hidden = my_gru.init_hidden()  # 形状: [1, 1, 128]

        # 7.3 前向传播.
        output, hidden = my_gru(x[0], hidden)  # x[0] 等价于: [10, 57]
        print(f'GRU输出形状: {output.shape}, 预测结果: {output}')

        # 扩展: 只训练1个样本, 不然太多了, 这里看看即可.
        if i == 0:
            break

# 模型训练.
# 定义变量, 记录: 学习率, 训练的轮数.
my_lr, epochs = 1e-3, 1

# RNN模型训练.
def train_rnn():
    # 1. 数据准备动作.
    # 1.1 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象.
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 2. 模型与优化器初始化.
    # 2.1 定义模型参数,
    # 参1: 输入维度(字符表大小), 参2: 隐藏层维度, 参3: 输出维度(国家数量)
    input_size, n_hidden, output_size = n_letters, 128, category_num       # 等价于: 57, 128, 18

    # 2.2 创建模型对象.
    my_rnn = My_RNN(input_size, n_hidden, output_size)

    # 2.3 定义损失函数和优化器.
    criterion = nn.NLLLoss()    # 如果你用了CrossEntropyLoss(), 则它 = NLLLoss() + LogSoftmax()
    optimizer = optim.Adam(my_rnn.parameters(), lr=my_lr)

    # 3. 训练过程 -> 参数初始化
    start_time = time.time()        # 模型开始训练时间
    total_iter_num = 0              # 已训练的样本数
    total_loss = 0.0                # 已训练的损失和
    total_loss_list = []            # 每100个样本求一次平均损失, 形成: 损失列表
    total_acc_num = 0               # 已训练的样本, 预测准确总数
    total_acc_list = []             # 每100个样本求一次平均准确率, 形成: 准确率列表

    # 4. 具体的训练过程, 按轮数遍历数据集.
    for epoch in range(epochs):     # epoch: 第几轮

        print(f'\n开始第{epoch + 1}/{epochs} 轮训练...')

        # 4.1 创建数据集加载器对象, 随机打乱数据集.
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

        # 4.2 样本迭代训练,  即: 本轮具体的每批次训练
        for i, (x, y) in enumerate(tqdm(train_dataloader)):     # 优化点3: 这里加入进度条.
            # 4.3 前向传播, 计算结果.
            output, hidden = my_rnn(x[0], my_rnn.init_hidden())
            # 4.4 计算损失.
            my_loss = criterion(output, y)
            # 4.5 三剑客 -> 梯度清零, 反向传播, 优化器更新参数.
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 4.6 统计训练结果(指标统计)
            total_iter_num += 1             # 训练的样本数 + 1
            total_loss += my_loss.item()    # 累计损失值

            # 4.7 计算当前样本预测准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y else 0)        # 统计: 预测正确的样本数

            # 4.8 统计: 每100个样本求一次平均损失, 准确率 形成: 损失列表, 准确率列表.
            if total_iter_num % 100 == 0:
                # 走这里, 说明100步了, 计算: 平均损失.
                avg_loss = total_loss / total_iter_num      # 总损失 / 总样本数
                # 把上述的平均损失, 添加到: 损失列表.
                total_loss_list.append(avg_loss)

                # 计算准确率, 即: 预测正确的 / 总样本数, 并添加到: 准确率列表.
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 4.9 每2000步(个样本), 打印训练日志.
            if total_iter_num % 2000 == 0:
                # 计算平均损失.
                avg_loss = total_loss / total_iter_num
                # 计算模型训练耗时
                end_time = int(time.time() - start_time)
                # 输出训练日志.
                print(f'轮次: {epoch + 1}, 训练的样本数: {total_iter_num}, 平均损失: {avg_loss:.4f}, 耗时: {end_time}s, 准确率: {avg_acc:.4f}')

        # 4.10 走到这里, 说明一轮训练完毕 -> 保存模型.
        torch.save(my_rnn.state_dict(), f'./model/my_rnn_GNC_{epoch + 1}.bin')

    # 5. 走到这里, 训练结束, 返回统计结果.
    total_time = int(time.time() - start_time)
    print(f'训练完成, 总耗时: {total_time}s, 总训练了 {total_iter_num}个样本!!')

    # 6. 优化4: 你可以把下述返回的三个值(损失列表, 训练总耗时, 准确率列表), 存储到文件中.
    #          因为一会儿我们会 可视化3个模型的训练结果, 如果没有存储的话, 会把 训练动作从新跑一次.

    # 7. 返回结果: 损失列表, 训练总耗时, 准确率列表.
    return total_loss_list, total_time, total_acc_list

# LSTM模型训练.
def train_lstm():
    # 1. 数据准备动作.
    # 1.1 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象.
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 2. 模型与优化器初始化.
    # 2.1 定义模型参数,
    # 参1: 输入维度(字符表大小), 参2: 隐藏层维度, 参3: 输出维度(国家数量)
    input_size, n_hidden, output_size = n_letters, 128, category_num       # 等价于: 57, 128, 18

    # 2.2 创建模型对象.
    my_rnn = My_LSTM(input_size, n_hidden, output_size)

    # 2.3 定义损失函数和优化器.
    criterion = nn.NLLLoss()    # 如果你用了CrossEntropyLoss(), 则它 = NLLLoss() + LogSoftmax()
    optimizer = optim.Adam(my_rnn.parameters(), lr=my_lr)

    # 3. 训练过程 -> 参数初始化
    start_time = time.time()        # 模型开始训练时间.
    total_iter_num = 0              # 已训练的样本数.
    total_loss = 0.0                # 已训练的损失和
    total_loss_list = []            # 每100个样本求一次平均损失, 形成: 损失列表.
    total_acc_num = 0               # 已训练的样本, 预测准确总数
    total_acc_list = []             # 每100个样本求一次平均准确率, 形成: 准确率列表.

    # 4. 具体的训练过程, 按轮数遍历数据集.
    for epoch in range(epochs):     # epoch: 第几轮
        print(f'\n开始第{epoch + 1}/{epochs} 轮训练...')
        # 4.1 创建数据集加载器对象, 随机打乱数据集.
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
        # 4.2 样本迭代训练,  即: 本轮具体的每批次训练
        for i, (x, y) in enumerate(tqdm(train_dataloader)):     # 优化点3: 这里加入进度条.
            # 4.3 前向传播, 计算结果.
            hidden, c = my_rnn.init_hidden()
            output, hidden, c = my_rnn(x[0], hidden, c)
            # 4.4 计算损失.
            my_loss = criterion(output, y)
            # 4.5 三剑客 -> 梯度清零, 反向传播, 优化器更新参数.
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 4.6 统计训练结果(指标统计)
            total_iter_num += 1             # 训训练的样本数 + 1
            total_loss += my_loss.item()    # 累计损失值

            # 4.7 计算当前样本预测准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y else 0)        # 统计: 预测正确的样本数

            # 4.8 统计: 每100个样本求一次平均损失, 准确率 形成: 损失列表, 准确率列表.
            if total_iter_num % 100 == 0:
                # 走这里, 说明100步了, 计算: 平均损失.
                avg_loss = total_loss / total_iter_num      # 总损失 / 总样本数
                # 把上述的平均损失, 添加到: 损失列表.
                total_loss_list.append(avg_loss)

                # 计算准确率, 即: 预测正确的 / 总样本数, 并添加到: 准确率列表.
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 4.9 每2000步(个样本), 打印训练日志.
            if total_iter_num % 2000 == 0:
                # 计算平均损失.
                avg_loss = total_loss / total_iter_num
                # 计算模型训练耗时
                end_time = int(time.time() - start_time)
                # 输出训练日志.
                print(f'轮次: {epoch + 1}, 训练的样本数: {total_iter_num}, 平均损失: {avg_loss:.4f}, 耗时: {end_time}s, 准确率: {avg_acc:.4f}')

        # 4.10 走到这里, 说明一轮训练完毕 -> 保存模型.
        torch.save(my_rnn.state_dict(), f'./model/my_lstm_GNC_{epoch + 1}.bin')

    # 5. 走到这里, 训练结束, 返回统计结果.
    total_time = int(time.time() - start_time)
    print(f'训练完成, 总耗时: {total_time}s, 总训练了 {total_iter_num}个样本!!')

    # 6. 优化4: 你可以把下述返回的三个值(损失列表, 训练总耗时, 准确率列表), 存储到文件中.
    #          因为一会儿我们会 可视化3个模型的训练结果, 如果没有存储的话, 会把 训练动作从新跑一次.

    # 7. 返回结果: 损失列表, 训练总耗时, 准确率列表.
    return total_loss_list, total_time, total_acc_list

# GRU模型训练.
def train_gru():
    # 1. 数据准备动作.
    # 1.1 读取数据
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # 1.2 构建数据集对象.
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 2. 模型与优化器初始化.
    # 2.1 定义模型参数,
    # 参1: 输入维度(字符表大小), 参2: 隐藏层维度, 参3: 输出维度(国家数量)
    input_size, n_hidden, output_size = n_letters, 128, category_num       # 等价于: 57, 128, 18

    # 2.2 创建模型对象.
    my_rnn = My_GRU(input_size, n_hidden, output_size)

    # 2.3 定义损失函数和优化器.
    criterion = nn.NLLLoss()    # 如果你用了CrossEntropyLoss(), 则它 = NLLLoss() + LogSoftmax()
    optimizer = optim.Adam(my_rnn.parameters(), lr=my_lr)

    # 3. 训练过程 -> 参数初始化
    start_time = time.time()        # 模型开始训练时间.
    total_iter_num = 0              # 已训练的样本数.
    total_loss = 0.0                # 已训练的损失和
    total_loss_list = []            # 每100个样本求一次平均损失, 形成: 损失列表.
    total_acc_num = 0               # 已训练的样本, 预测准确总数
    total_acc_list = []             # 每100个样本求一次平均准确率, 形成: 准确率列表.

    # 4. 具体的训练过程, 按轮数遍历数据集.
    for epoch in range(epochs):     # epoch: 第几轮
        print(f'\n开始第{epoch + 1}/{epochs} 轮训练...')
        # 4.1 创建数据集加载器对象, 随机打乱数据集.
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)
        # 4.2 样本迭代训练,  即: 本轮具体的每批次训练
        for i, (x, y) in enumerate(tqdm(train_dataloader)):     # 优化点3: 这里加入进度条.
            # 4.3 前向传播, 计算结果.
            output, hidden = my_rnn(x[0], my_rnn.init_hidden())
            # 4.4 计算损失.
            my_loss = criterion(output, y)
            # 4.5 三剑客 -> 梯度清零, 反向传播, 优化器更新参数.
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 4.6 统计训练结果(指标统计)
            total_iter_num += 1             # 训训练的样本数 + 1
            total_loss += my_loss.item()    # 累计损失值

            # 4.7 计算当前样本预测准确率
            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y else 0)        # 统计: 预测正确的样本数

            # 4.8 统计: 每100个样本求一次平均损失, 准确率 形成: 损失列表, 准确率列表.
            if total_iter_num % 100 == 0:
                # 走这里, 说明100步了, 计算: 平均损失.
                avg_loss = total_loss / total_iter_num      # 总损失 / 总样本数
                # 把上述的平均损失, 添加到: 损失列表.
                total_loss_list.append(avg_loss)

                # 计算准确率, 即: 预测正确的 / 总样本数, 并添加到: 准确率列表.
                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 4.9 每2000步(个样本), 打印训练日志.
            if total_iter_num % 2000 == 0:
                # 计算平均损失.
                avg_loss = total_loss / total_iter_num
                # 计算模型训练耗时
                end_time = int(time.time() - start_time)
                # 输出训练日志.
                print(f'轮次: {epoch + 1}, 训练的样本数: {total_iter_num}, 平均损失: {avg_loss:.4f}, 耗时: {end_time}s, 准确率: {avg_acc:.4f}')

        # 4.10 走到这里, 说明一轮训练完毕 -> 保存模型.
        torch.save(my_rnn.state_dict(), f'./model/my_gru_GNC_{epoch + 1}.bin')

    # 5. 走到这里, 训练结束, 返回统计结果.
    total_time = int(time.time() - start_time)
    print(f'训练完成, 总耗时: {total_time}s, 总训练了 {total_iter_num}个样本!!')

    # 6. 优化4: 你可以把下述返回的三个值(损失列表, 训练总耗时, 准确率列表), 存储到文件中.
    #          因为一会儿我们会 可视化3个模型的训练结果, 如果没有存储的话, 会把 训练动作从新跑一次.


    # 7. 返回结果: 损失列表, 训练总耗时, 准确率列表.
    return total_loss_list, total_time, total_acc_list


# 模型训练绘图 -> 这个函数的可视化代码你可以不写, 但是模型训练等代码要写出来, 出3张图.
def test_train_rnn_lstm_gru():
    # 1. 训练3种模型, 并获取性能指标.
    # 参1: 损失列表, 参2: 训练总耗时, 参3: 准确率列表.
    total_loss_list_rnn, total_time_rnn, total_acc_list_rnn = train_rnn()
    total_loss_list_lstm, total_time_lstm, total_acc_list_lstm = train_lstm()
    total_loss_list_gru, total_time_gru, total_acc_list_gru = train_gru()

    # 2. 绘制 损失对比曲线(评估: 模型收敛速度)
    # 2.1 创建画布.     0: 图1
    plt.figure(0, figsize=(10, 5))
    # 2.2 绘制各模型损失曲线.
    plt.plot(total_loss_list_rnn, label='RNN')
    plt.plot(total_loss_list_lstm, label='LSTM')
    plt.plot(total_loss_list_gru, label='GRU')
    # 2.3 设置图表属性
    plt.title('模型损失对比曲线')
    plt.xlabel('训练步数(每100步)')
    plt.ylabel('平均损失值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_loss_time.png')
    plt.show()

    # 3. 绘制 训练耗时对比柱状图(评估: 模型计算效率)
    # 3.1 创建画布,     1: 图2
    plt.figure(1, figsize=(10, 5))
    # 3.2 准备x轴 和 y轴标签内容.
    x_data = ['RNN', 'LSTM', 'GRU']
    y_data = [total_time_rnn, total_time_lstm, total_time_gru]

    # 3.3 绘制柱状图
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)

    # 3.4 设置图表属性
    plt.title('模型耗时对比柱状图')
    plt.savefig('./img/RNN_LSTM_GRU_time.png')
    plt.show()


    # 4. 绘制 训练准确率对比曲线(评估: 模型效果)
    # 4.1 创建画布.     2: 图3
    plt.figure(2, figsize=(10, 5))
    # 4.2 绘制各模型准确率曲线.
    plt.plot(total_acc_list_rnn, label='RNN', color='red')
    plt.plot(total_acc_list_lstm, label='LSTM', color='green')
    plt.plot(total_acc_list_gru, label='GRU', color='orange')
    # 4.3 绘制图表属性.
    plt.title('模型准确率对比曲线')
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_acc.png')
    plt.show()


# todo 10. 模型预测.
# todo 10.1 RNN模型预测
# todo 10.1.1. 定义遍历, 记录模型的参数的路径.
my_rnn_path = './model/my_rnn_GNC_1.bin'
my_lstm_path = './model/my_lstm_GNC_1.bin'
my_gru_path = './model/my_gru_GNC_1.bin'

# todo 10.1.2 定义函数, 将要预测的人名 转成 one-hot编码, 例如: 'zhang' -> [5, 57]
def lineToTensor(line):
    # 1. 初始化张量, [文本长度, 字符表长度]
    tensor_x = torch.zeros(len(line), n_letters)

    # 2. 遍历文本, 获取到每个字符及其索引.
    for i, letter in enumerate(line):
        # 3. 查看字符在全局字母表中的位置(索引)
        letter_index = all_letters.find(letter)
        # 4. 在张量的对应位置改为1, 完成: one-hot编码
        tensor_x[i][letter_index] = 1

    # 5. 返回结果.
    return tensor_x         # 即: 'zhang' -> [5, 57]


# todo 10.1.3 定义函数, 实现: RNN预测.
def predict_rnn(x):
    # 1. 定义遍历, 记录模型相关参数.
    n_letters, n_hidden, n_categories = 57, 128, 18
    # 2. 把输入的文字转成 one-hot编码.
    x_tensor = lineToTensor(x)
    # 3. 创建模型对象.
    my_rnn = My_RNN(n_letters, n_hidden, n_categories)
    # 4. 加载模型参数.
    my_rnn.load_state_dict(torch.load(my_rnn_path))
    # 5. 进行预测, 不计算梯度.  -> 节省内存和计算机资源.
    with torch.no_grad():
        # 5.1 模型预测
        output, hidden = my_rnn(x_tensor, my_rnn.init_hidden())
        # 5.2 从预测结果中, 获取前3个最大的元素.
        # 参1(k):   取前3个最大的元素.
        # 参2(dim): 获取概率最大的元素所在的维度.
        # 参3(largest): 获取概率最大的元素.
        topv, topi = output.topk(3, 1, True)
        # 5.3 打印待预测文本.
        print(f'rnn(待预测文本): {x}')

        # 5.4 解析预测结果.
        for i in range(3):
            value = topv[0][i].item()           # 概率值 -> Python的标量
            category_idx  = topi[0][i].item()   # 类别索引
            category = categories[category_idx] # 类别名称.
            print(f'value: {value}, category: {category}')



if __name__ == '__main__':
    # 1. 读取数据
    # my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 2. 测试: 数据加载器.
    # get_dataloader()

    # 3. 测试RNN模型
    # dm_test_myrnn()

    # 4. 测试RNN, LSTM, GRU模型
    # test_rnn_lstm_gru()

    # 5. 测试: 模型训练.
    # train_rnn()       
    # train_lstm()       
    # train_gru()         

    # 6. 测试: 模型训练绘图(即: 效果对比)
    # test_train_rnn_lstm_gru()

    # 7. 测试: 模型预测.
    predict_rnn('Piao')
```





# 三、注意力机制

## 3.1 注意力机制介绍

### 3.1.1 概念

**注意力机制**（Attention Mechanism）借鉴了人类视觉注意力的思想，在深度学习中用于让模型在处理信息时能够**动态地聚焦于输入数据中更重要的部分**。在自然语言处理中，注意力机制允许模型在处理某个词或生成某个输出时，为输入序列中的不同词分配不同的权重，从而捕捉序列中长距离依赖关系。 

数学上，给定一组输入向量（如编码器所有时间步的隐藏状态）和一个查询向量（query），注意力机制计算查询与每个输入之间的相关性得分，再通过 softmax 归一化为概率分布（注意力权重），最后对所有输入向量进行加权求和，得到上下文向量（context vector）。

### 3.1.2 为什么需要注意力

传统 Seq2Seq 模型（如基于 RNN 的 Encoder-Decoder）将整个输入序列压缩为一个固定长度的上下文向量，存在以下问题：

- **信息瓶颈**：无论输入多长，上下文向量维度固定，难以完整保留所有信息。
- **长距离依赖困难**：RNN 本身存在梯度消失问题，解码器难以获取输入序列早期的重要信息。
- **缺乏可解释性**：模型无法明确指出输出时关注了输入的哪些部分。

注意力机制通过**动态加权**输入序列，使解码器在每个时间步都能**直接访问**编码器的所有隐藏状态，有效缓解了上述问题，并提高了模型性能和可解释性。

### 3.1.3 注意力QKV

注意力机制通常使用三个概念：**查询（Query）**、**键（Key）**、**值（Value）**，简称 QKV。这三者来源于信息检索的类比：

- **Query（Q）查询向量**：表示当前需要关注的目标，当前要查询/解决的问题。例如解码器当前时间步的隐藏状态。
- **Key（K）键向量**：表示输入序列中每个元素的“索引”，用于匹配问题的索引，例如编码器各时间步的隐藏状态。
- **Value（V）值向量**：表示输入序列中每个元素实际包含的信息，最终获取的实际内容，通常与 Key 相同或由 Key 变换得到。

注意力计算过程可以概括为：根据 Query 和各个 Key 的**相似度**，得到注意力权重，再用这些权重对 Value 进行加权求和。

### 3.1.4 注意力机制步骤

通用的注意力计算步骤如下：

1. **计算相似度得分**：用Q和KEY进行相似度计算，得到一个注意力机制的权重分布。对于每个键 $k_i$，计算查询 $q$ 与它的相似度得分 $e_i$。
   常用方法：
   
   - 点积（Dot Product）：$e_i = q \cdot k_i$，计算匹配度
   - 缩放点积（Scaled Dot Product）：$e_i = \frac{q \cdot k_i}{\sqrt{d_k}}$，其中 $d_k$ 是键的维度
   - 加性注意力（Additive）：$e_i = v^T \tanh(W_1 q + W_2 k_i)$
   
2. **归一化得到注意力权重**：使用 softmax 将得分转换为概率分布。
   $$
   \alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}
   $$

3. **加权求和得到上下文向量**：
   $$
   c = \sum_i \alpha_i v_i
   $$

在自注意力（Self-Attention）中，Query、Key、Value 均来自同一输入序列的不同线性变换。

---

## 3.2 Seq2Seq架构中的注意力机制

### 3.2.1 Seq2Seq架构

Seq2Seq（Sequnce-to-Sequence）模型由编码器（Encoder）和解码器（Decoder）组成，用于处理输入输出长度不等的序列转换任务，如机器翻译、文本摘要、对话系统等任务。

- **编码器**：读取输入序列，将输入序列**映射成一个中间的表示**（中间语义张量c），每个时间步都会生成一个隐藏状态 $h_1, h_2, \dots, h_T$。最后一个时间步输出的隐藏状态会作为解码器的初始隐藏状态。常用编码器结构有RNN、LSTM、Transformer等
- **解码器**：逐步生成输出序列，每个时间步 $t$ 都会用到中间语义张量c，产生一个输出 $y_t$，其隐藏状态为 $s_t$。

在不使用注意力的 Seq2Seq 中，解码器只依赖编码器最后一个隐藏状态作为初始状态，或将其作为固定上下文向量。

加入注意力后，解码器在每一步都会动态计算一个上下文向量 $c_t$，该向量是编码器所有隐藏状态的加权和，权重由当前解码器状态 $s_t$ 与编码器隐藏状态 $h_i$ 的相似度决定。

![image-20260907134443193](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260907134443193.png)

v是输入词的词向量，h、s都表示隐藏状态，c表示中间语义张量 可以理解为一个总信息包，

编码流程：逐个时间步进行编码，每个时间步都有隐藏层输出，最终组合成中间语义张量c

解码流程：逐个时间步进行解码，每个时间步输入input(c)、$\mathbf{h}_{t-1}$，输出output、$h_t$

### 3.2.2 Seq2Seq架构中的QKV

在 seq2seq 注意力机制中：

- **Query**：解码器当前时间步的隐藏状态 $s_t$（有时经过线性变换）。
- **Key**：编码器各时间步的隐藏状态 $h_i$（通常经过线性变换）。
- **Value**：通常与 Key 相同（或经过另一线性变换），即编码器隐藏状态 $h_i$。

具体计算时，先对 Query 和 Key 计算相似度得分，例如使用加性注意力或缩放点积，然后 softmax 得到注意力权重，最后对 Value 加权求和得到上下文向量 $c_t$。这个 $c_t$ 会与解码器状态 $s_t$ 结合，用于预测当前输出。

---

### 3.2.3 Seq2Seq架构原生痛点和改进

#### 3.2.3.1 Seq2Seq架构原生痛点

传统 Seq2Seq 模型（基于 RNN 的 Encoder-Decoder）存在以下主要问题：

1. **固定长度上下文向量**的信息瓶颈，编码器将整个输入序列压缩为一个固定长度的上下文向量c（通常是编码器最后一个时间步的隐藏状态）。无论输入序列多长，解码器只能从这个固定向量中获取信息。当输入序列较长时，这个向量难以保留所有细节，导致信息丢失。

2. **长距离依赖问题**，RNN 本身存在梯度消失/爆炸问题，难以捕捉输入序列中远距离的依赖关系。即使使用 LSTM 或 GRU，对于非常长的序列，早期输入的信息在传递到编码器最后状态时可能已经被稀释。

3. **缺乏对齐能力**传统 Seq2Seq 模型无法显式地表示输入和输出之间的对齐关系。例如在机器翻译中，输出词往往对应输入序列中某个特定的词或短语，但模型只能通过固定向量隐式地传递这种对齐信息，可解释性差。

4. 解码器每个时间步使用相同的上下文向量在传统架构中，所有解码时间步使用同一个上下文向量。但实际上，生成不同输出词时，模型应该关注输入的不同部分。使用相同向量限制了模型的表达能力。



#### 3.2.3.2 改进方法：引入注意力机制

注意力机制是 Seq2Seq 架构最重要的改进之一，有效缓解了上述痛点。

1. **动态上下文向量注意力机制**允许解码器在每个时间步动态计算一个上下文向量，该向量是编码器所有隐藏状态的加权和。权重根据当前解码器状态与编码器各隐藏状态的相似度确定，因此不同时间步的上下文向量不同，能够聚焦于输入的不同部分。

2. **直接访问**编码器所有状态解码器不再只依赖编码器的最后一个隐藏状态，而是可以访问编码器所有时间步的隐藏状态。这样，即使输入序列很长，模型也能直接“看到”早期信息，缓解了信息瓶颈和长距离依赖问题。

3. **对齐可视化**注意力权重可以直观地展示输入和输出之间的对齐关系，提高了模型的可解释性。通过分析注意力矩阵，可以观察模型在生成某个输出词时关注了输入的哪些词。

4. **训练更容易**注意力机制的引入使得梯度可以更直接地流回编码器，避免了长距离梯度传播的问题，模型更容易训练。





### 3.2.4 引入注意力机制

传统 Seq2Seq 模型将整个输入序列压缩为固定长度的上下文向量，导致信息瓶颈。注意力机制通过让解码器在每个时间步动态关注输入序列的不同部分，有效缓解了这一问题。引入注意力后，解码器不再仅仅依赖编码器最后一个隐藏状态，而是利用编码器所有时间步的隐藏状态，根据当前解码状态计算相关性权重，生成动态的上下文向量。

#### 3.2.4.1注意力机制在 Seq2Seq 中的工作流程

在基于 RNN 的 Seq2Seq 中加入注意力，核心步骤为：

1. **编码器**：输入序列经过双向 RNN（或单向 RNN）得到每个时间步的隐藏状态 $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T$，每个token都对应一个隐藏状态 $\mathbf{h}_i$
2. **解码器**：在每个解码时间步 $t$，使用当前解码器隐藏状态 $\mathbf{s}_t$ 作为 Query，编码器所有隐藏状态 $\mathbf{h}_i$ 作为 Key 和 Value。
3. **计算注意力权重**：
   - 计算 Query 与每个 Key 的相关性得分 $e_{ti}$。
   - 使用 softmax 归一化为注意力权重 $\alpha_{ti}$。
4. **计算上下文向量**：对所有 Value 加权求和得到 $\mathbf{c}_t$。
5. **生成输出**：将上下文向量 $\mathbf{c}_t$ 与解码器状态 $\mathbf{s}_t$ 拼接，经过全连接层和 softmax 预测下一个词。

![image-20260907163511596](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260907163511596.png)



![image-20260907164215220](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260907164215220.png)



#### 3.2.4.2 注意力计算公式

注意力计算公式变量说明表

|             变量名              |                         变量含义                         |                             解释                             |
| :-----------------------------: | :------------------------------------------------------: | :----------------------------------------------------------: |
|            $e_{ti}$             | 第 $t$ 个解码时间步对第 $i$ 个编码器隐藏状态的注意力得分 | 衡量解码器当前状态与输入位置 $i$ 的相关性。在加性注意力中通过 $\mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_t + \mathbf{W}_h \mathbf{h}_i + \mathbf{b})$ 计算；在点积注意力中通过 $\mathbf{s}_t^\top \mathbf{h}_i$ 或 $\frac{\mathbf{s}_t^\top \mathbf{h}_i}{\sqrt{d}}$ 计算 |
|          $\mathbf{v}$           |                   注意力得分向量的参数                   |       仅用于加性注意力，将 $\tanh$ 输出映射为标量得分        |
|         $\mathbf{W}_s$          |                   解码器状态的权重矩阵                   | 仅用于加性注意力，对当前解码器隐藏状态 $\mathbf{s}_t$ 进行线性变换 |
|         $\mathbf{W}_h$          |                 编码器隐藏状态的权重矩阵                 | 仅用于加性注意力，对编码器隐藏状态 $\mathbf{h}_i$ 进行线性变换 |
|          $\mathbf{b}$           |                    注意力得分偏置向量                    |            仅用于加性注意力，调整非线性变换的输出            |
|         $\mathbf{s}_t$          |                解码器当前时间步的隐藏状态                |   作为 Query，表示当前需要关注的目标信息，两种注意力均使用   |
|         $\mathbf{h}_i$          |             编码器第 $i$ 个时间步的隐藏状态              | 作为 Key（和 Value），提供输入序列中位置 $i$ 的信息，两种注意力均使用 |
|               $d$               |                       隐藏状态维度                       |           仅用于缩放点积注意力，作为缩放因子的依据           |
|           $\sqrt{d}$            |                         缩放因子                         |    仅用于缩放点积注意力，对点积结果进行缩放，防止数值过大    |
|          $\alpha_{ti}$          |    第 $t$ 个解码时间步对第 $i$ 个输入位置的注意力权重    | 对 $e_{ti}$ 进行 softmax 归一化后的概率分布，所有 $\alpha_{ti}$ 之和为 1，两种注意力均使用 |
|         $\mathbf{c}_t$          |             第 $t$ 个解码时间步的上下文向量              | 编码器隐藏状态的加权和 $\sum_i \alpha_{ti} \mathbf{h}_i$，汇聚了当前关注的输入信息，两种注意力均使用 |
|     $\tilde{\mathbf{s}}_t$      |               解码器融合上下文后的中间状态               | 仅用于加性注意力，由 $[\mathbf{s}_t ; \mathbf{c}_t]$ 经线性变换和 $\tanh$ 得到 |
|         $\mathbf{W}_c$          |                上下文与状态融合的权重矩阵                |         仅用于加性注意力，对拼接后的向量进行线性变换         |
|         $\mathbf{b}_c$          |                     融合层的偏置向量                     |              仅用于加性注意力，调整融合层的输出              |
|       $\mathbf{W}_{out}$        |                       输出投影矩阵                       |    仅用于加性注意力，将融合状态映射到词汇表大小的 logits     |
|       $\mathbf{b}_{out}$        |                       输出偏置向量                       |              仅用于加性注意力，调整输出 logits               |
|             $\tanh$             |                     双曲正切激活函数                     | 在加性注意力中用于非线性变换，在融合层中也使用，将值压缩到 $[-1,1]$ |
|        $\text{softmax}$         |                       Softmax 函数                       |           将注意力得分或输出 logits 转换为概率分布           |
|               $T$               |                       输入序列长度                       |               编码器时间步总数，用于求和归一化               |
| $[\mathbf{s}_t ; \mathbf{c}_t]$ |                         拼接操作                         |   仅用于加性注意力，将解码器状态和上下文向量拼接为一个向量   |

以加性注意力（Additive Attention / Bahdanau Attention）为例，计算公式如下：

1. **相关性得分**：
   $$
   e_{ti} = \mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_t + \mathbf{W}_h \mathbf{h}_i + \mathbf{b})
   $$
   **变量来源与作用：**

   - **$\mathbf{s}_t$**：解码器当前时间步 $t$ 的隐藏状态，来源于解码器 RNN 的前向计算。它表示当前已经生成的信息，是查询向量（Query），用于确定需要关注输入序列的哪些部分。
   - **$\mathbf{h}_i$**：编码器在输入位置 $i$ 的隐藏状态，来源于编码器 RNN 的输出。它表示输入序列中第 $i$ 个词及其上下文的信息，作为键（Key）和值（Value）。
   - **$\mathbf{W}_s$** 和 **$\mathbf{W}_h$**：可学习的权重矩阵。它们分别对 $\mathbf{s}_t$ 和 $\mathbf{h}_i$ 进行线性变换，将两者映射到同一个向量空间，以便进行相加和比较。
   - **$\mathbf{b}$**：可学习的偏置向量，用于调整线性组合后的输出。
   - **$\tanh$**：双曲正切激活函数，将线性组合的结果压缩到 $[-1,1]$，引入非线性，使模型能够学习更复杂的相关性模式。
   - **$\mathbf{v}$**：可学习的参数向量，将 $\tanh$ 输出的向量映射为一个标量得分。实际上，$\mathbf{v}^\top \cdot \tanh(\cdot)$ 等价于一个单层神经网络，将高维表示转换为一个实数。
   
   **公式作用：**  
   计算解码器状态 $\mathbf{s}_t$ 与编码器隐藏状态 $\mathbf{h}_i$ 之间的相关性得分 $e_{ti}$。得分越高，表示当前解码步骤应该更多地关注输入位置 $i$。这个得分是后续 softmax 归一化的基础。
   
   
   
2. **注意力权重**：
   $$
   \alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^T \exp(e_{tj})}
   $$
   **变量来源与作用：**
   
   - **$e_{ti}$**：由上一步计算得到的相关性得分。
   - **$\exp$**：指数函数，将得分映射为正数，并放大差异。
   - **分母 $\sum_{j=1}^T \exp(e_{tj})$**：对所有输入位置 $j$ 的指数得分求和，作为归一化因子。
   
   **公式作用：**  
   将注意力得分转换为概率分布 $\alpha_{ti}$，满足 $\sum_{i=1}^T \alpha_{ti} = 1$ 且 $\alpha_{ti} \geq 0$。该分布表示在生成第 $t$ 个输出时，输入序列中每个位置的重要性权重。softmax 操作确保了权重的可解释性和稳定性。
   
   
   
3. **上下文向量**：
   $$
   \mathbf{c}_t = \sum_{i=1}^T \alpha_{ti} \mathbf{h}_i
   $$
   **变量来源与作用：**
   
   - **$\alpha_{ti}$**：注意力权重，由 softmax 得到。
   - **$\mathbf{h}_i$**：编码器在位置 $i$ 的隐藏状态，作为值（Value）。
   - **$\sum$**：对所有输入位置加权求和。
   
   **公式作用：**  
   将输入序列的所有隐藏状态按注意力权重进行加权平均，得到当前解码时间步 $t$ 的上下文向量 $\mathbf{c}_t$。该向量动态地汇总了与当前解码最相关的输入信息，是解码器生成输出时的主要依据。注意力权重越高，对应的 $\mathbf{h}_i$ 在 $\mathbf{c}_t$ 中的贡献越大。
   
   
   
4. **输出预测**：
   $$
   \tilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{s}_t ; \mathbf{c}_t] + \mathbf{b}_c)
   $$

**变量来源与作用：**

- **$\mathbf{s}_t$**：解码器当前隐藏状态。
- **$\mathbf{c}_t$**：由上一步计算的上下文向量。
- **$[\mathbf{s}_t ; \mathbf{c}_t]$**：拼接操作，将解码器状态和上下文向量拼接为一个更长的向量，使模型同时考虑当前解码状态和输入注意力信息。
- **$\mathbf{W}_c$**：可学习的权重矩阵，对拼接后的向量进行线性变换。
- **$\mathbf{b}_c$**：偏置向量。
- **$\tanh$**：激活函数，引入非线性并压缩到 $[-1,1]$。

**公式作用：**  
将解码器自身状态与上下文向量融合，生成一个综合表示 $\tilde{\mathbf{s}}_t$。这个表示同时包含了当前已生成内容的信息和输入序列中对齐的信息，为下一步预测词汇提供更丰富的特征。


$$
p(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_{out} \tilde{\mathbf{s}}_t + \mathbf{b}_{out})
$$
**变量来源与作用：**

- **$\tilde{\mathbf{s}}_t$**：融合后的状态，由上一个公式得到。
- **$\mathbf{W}_{out}$**：输出投影矩阵，将 $\tilde{\mathbf{s}}_t$ 映射到词汇表大小的向量（logits）。
- **$\mathbf{b}_{out}$**：输出偏置向量。
- **$\text{softmax}$**：将 logits 转换为概率分布，表示在词汇表中每个词作为当前输出 $y_t$ 的概率。

**公式作用：**  

预测当前时间步输出词的条件概率分布。给定之前生成的词 $y_{<t}$ 和输入序列 $\mathbf{x}$，模型输出下一个词的概率。训练时通常使用交叉熵损失函数进行优化。



另一种常见形式是点积注意力（Luong Attention）：
$$
e_{ti} = \mathbf{s}_t^\top \mathbf{h}_i
$$
**变量来源与作用：**

- **$\mathbf{s}_t$**：解码器当前隐藏状态，作为查询（Query）。
- **$\mathbf{h}_i$**：编码器隐藏状态，作为键（Key）。
- **$\mathbf{s}_t^\top \mathbf{h}_i$**：两个向量的点积，直接衡量它们之间的相似度（余弦相似度未归一化）。点积越大，表示两个向量方向越一致，相关性越高。

**公式作用：**  
计算解码器状态与编码器每个位置之间的相似度得分。相比加性注意力，点积计算更简单、高效，但当维度较大时数值可能过大，导致 softmax 梯度消失，因此常使用缩放点积。



或缩放点积：
$$
e_{ti} = \frac{\mathbf{s}_t^\top \mathbf{h}_i}{\sqrt{d}}
$$
其中 $d$ 为隐藏状态维度。

**变量来源与作用：**

- **$\mathbf{s}_t^\top \mathbf{h}_i$**：点积得分。
- **$d$**：隐藏状态的维度（即 $\mathbf{s}_t$ 和 $\mathbf{h}_i$ 的维度）。
- **$\sqrt{d}$**：缩放因子。当 $d$ 较大时，点积的方差会增大，导致 softmax 输出过于尖锐或梯度消失。除以 $\sqrt{d}$ 可以保持数值稳定，使梯度更易传播。

**公式作用：**  

对点积得分进行缩放，避免数值过大。缩放点积注意力是 Transformer 中自注意力机制的基础，具有计算高效、易于并行等优点。





## 3.3 注意力计算规则

### 3.3.1 加性注意力（Additive Attention）

#### 3.3.1.1 公式

加性注意力（又称 Bahdanau Attention）使用一个前馈神经网络来计算查询（Query）和键（Key）之间的相关性得分。  
给定查询 $\mathbf{q}$ 和键 $\mathbf{k}_i$，其得分 $e_i$ 计算如下：

$$
e_i = \mathbf{v}^\top \tanh(\mathbf{W}_q \mathbf{q} + \mathbf{W}_k \mathbf{k}_i + \mathbf{b})
$$

随后通过 softmax 得到注意力权重：

$$
\alpha_i = \frac{\exp(e_i)}{\sum_{j} \exp(e_j)}
$$

最终上下文向量 $\mathbf{c}$ 为所有值（Value）的加权和（通常值等于键或键的线性变换）：

$$
\mathbf{c} = \sum_i \alpha_i \mathbf{v}_i
$$

#### 3.3.1.2 推导过程

注意力机制的核心思想是：给定一个查询 $\mathbf{q}$，从一组键值对 $(\mathbf{k}_i, \mathbf{v}_i)$ 中选择相关信息。加性注意力通过以下步骤实现：

1. **线性变换**：将查询和键分别通过可学习的权重矩阵 $\mathbf{W}_q$ 和 $\mathbf{W}_k$ 映射到同一向量空间，加上偏置 $\mathbf{b}$，得到变换后的表示。
2. **非线性激活**：对两个线性变换之和施加 $\tanh$ 激活函数，引入非线性，使模型能够学习更复杂的相关性模式。
3. **标量投影**：通过参数向量 $\mathbf{v}$ 将非线性结果投影为一个标量得分 $e_i$。
4. **归一化**：使用 softmax 将得分转换为概率分布 $\alpha_i$，保证所有权重之和为 1。
5. **加权求和**：根据权重对值 $\mathbf{v}_i$ 进行加权求和，得到上下文向量 $\mathbf{c}$。

加性注意力的优势在于非线性和可学习的参数，使其能够灵活地建模查询和键之间的复杂关系，但计算开销相对较大。

#### 3.3.1.3 变量说明表

|     变量名     |     变量含义     |                      解释                       |
| :------------: | :--------------: | :---------------------------------------------: |
|  $\mathbf{q}$  |     查询向量     |   当前需要关注的目标表示，通常来自解码器状态    |
| $\mathbf{k}_i$ | 第 $i$ 个键向量  |  输入序列中第 $i$ 个位置的表示，用于与查询比较  |
| $\mathbf{v}_i$ | 第 $i$ 个值向量  | 输入序列中第 $i$ 个位置的实际信息，用于加权求和 |
| $\mathbf{W}_q$ |  查询的权重矩阵  |        对查询 $\mathbf{q}$ 进行线性变换         |
| $\mathbf{W}_k$ |   键的权重矩阵   |        对键 $\mathbf{k}_i$ 进行线性变换         |
|  $\mathbf{b}$  |     偏置向量     |              调整线性组合后的输出               |
|    $\tanh$     | 双曲正切激活函数 |         引入非线性，将值压缩到 $[-1,1]$         |
|  $\mathbf{v}$  |   得分投影向量   |           将非线性结果映射为标量得分            |
|     $e_i$      |    注意力得分    |          查询与第 $i$ 个键的相关性度量          |
|   $\alpha_i$   |    注意力权重    |           softmax 归一化后的概率分布            |
|  $\mathbf{c}$  |    上下文向量    |          所有值的加权和，汇聚相关信息           |
|     $\exp$     |     指数函数     |         将得分转换为正数，用于 softmax          |
|      $j$       |     求和索引     |                 遍历所有键值对                  |

---

### 3.3.2 点积注意力（Dot-Product Attention）

#### 3.3.2.1 公式

点积注意力直接使用查询 $\mathbf{q}$ 和键 $\mathbf{k}_i$ 的内积作为相似度得分：

$$
e_i = \mathbf{q}^\top \mathbf{k}_i
$$

注意力权重和上下文向量的计算与加性注意力相同：

$$
\alpha_i = \frac{\exp(e_i)}{\sum_{j} \exp(e_j)}
$$

$$
\mathbf{c} = \sum_i \alpha_i \mathbf{v}_i
$$

#### 3.3.2.2 推导过程

点积注意力的理论基础是向量的点积可以作为相似度的度量。在向量空间中，两个向量的点积越大，表示它们方向越一致，相关性越高。其计算过程为：

1. **计算点积**：对查询 $\mathbf{q}$ 和每个键 $\mathbf{k}_i$ 计算内积，得到得分 $e_i$。这一步没有可学习参数，计算高效。
2. **归一化**：使用 softmax 将点积得分转换为注意力权重 $\alpha_i$。
3. **加权求和**：根据权重对值 $\mathbf{v}_i$ 加权求和，得到上下文向量 $\mathbf{c}$。

点积注意力简单快速，但当向量维度 $d$ 较大时，点积结果的方差会增大，导致 softmax 进入饱和区，梯度变小，影响训练。因此通常需要缩放（见下一小节）。

#### 3.3.2.3 变量说明表

|             变量名             |    变量含义     |                       解释                        |
| :----------------------------: | :-------------: | :-----------------------------------------------: |
|          $\mathbf{q}$          |    查询向量     |              当前需要关注的目标表示               |
|         $\mathbf{k}_i$         | 第 $i$ 个键向量 |           输入序列中第 $i$ 个位置的表示           |
|         $\mathbf{v}_i$         | 第 $i$ 个值向量 |         输入序列中第 $i$ 个位置的实际信息         |
|             $e_i$              |   注意力得分    | $\mathbf{q}$ 与 $\mathbf{k}_i$ 的点积，表示相似度 |
|           $\alpha_i$           |   注意力权重    |            softmax 归一化后的概率分布             |
|          $\mathbf{c}$          |   上下文向量    |                  所有值的加权和                   |
| $\mathbf{q}^\top \mathbf{k}_i$ |    点积运算     |                计算两个向量的内积                 |
|             $\exp$             |    指数函数     |                   用于 softmax                    |
|              $j$               |    求和索引     |                  遍历所有键值对                   |

---



### 3.3.3 缩放点积注意力（Scaled Dot-Product Attention）

#### 3.3.3.1 公式

缩放点积注意力是对点积注意力的改进，通过除以 $\sqrt{d}$ 来缩放点积结果，其中 $d$ 是查询Q和键K的维度：

$$
e_i = \frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{d}}
$$

其余计算保持一致：

$$
\alpha_i = \frac{\exp(e_i)}{\sum_{j} \exp(e_j)}
$$

$$
\mathbf{c} = \sum_i \alpha_i \mathbf{v}_i
$$

#### 3.3.2.2 推导过程

当向量的维度 $d$ 较大时，点积 $\mathbf{q}^\top \mathbf{k}_i$ 的方差约为 $d$，因为假设每个分量相互独立且均值为 0、方差为 1，则点积的方差为 $d$。较大的方差会导致 softmax 函数的输入绝对值过大，使得梯度变得很小（softmax 的饱和区），从而减慢训练。  

缩放点积通过除以 $\sqrt{d}$，使得点积结果的方差重新变为 1，保持数值稳定：

- 若 $\mathbf{q}$ 和 $\mathbf{k}_i$ 的各分量独立同分布，均值为 0，方差为 1，则 $\mathbf{q}^\top \mathbf{k}_i$ 的方差为 $d$。
- 除以 $\sqrt{d}$ 后，方差变为 1。

因此，缩放点积注意力有效避免了因维度增加导致的梯度消失问题，成为 Transformer 等现代模型的标准选择。

#### 3.3.2.3 变量说明表

|             变量名             |    变量含义     |               解释                |
| :----------------------------: | :-------------: | :-------------------------------: |
|          $\mathbf{q}$          |    查询向量     |      当前需要关注的目标表示       |
|         $\mathbf{k}_i$         | 第 $i$ 个键向量 |   输入序列中第 $i$ 个位置的表示   |
|         $\mathbf{v}_i$         | 第 $i$ 个值向量 | 输入序列中第 $i$ 个位置的实际信息 |
|              $d$               |    向量维度     | 查询和键的维度，用于计算缩放因子  |
|           $\sqrt{d}$           |    缩放因子     | 对点积结果进行缩放，防止方差过大  |
|             $e_i$              |   注意力得分    |     缩放后的点积，表示相似度      |
|           $\alpha_i$           |   注意力权重    |    softmax 归一化后的概率分布     |
|          $\mathbf{c}$          |   上下文向量    |          所有值的加权和           |
| $\mathbf{q}^\top \mathbf{k}_i$ |    点积运算     |        计算两个向量的内积         |
|             $\exp$             |    指数函数     |           用于 softmax            |
|              $j$               |    求和索引     |          遍历所有键值对           |

---

以上三种注意力计算规则共同构成了现代注意力机制的基础，尤其缩放点积注意力在 Transformer 中发挥了核心作用。在实际应用中，可根据任务复杂度和计算需求选择合适的计算方式。



### 3.3.4 计算规则的实现

### 3.3.4.1 torch.bmm 的运算规则

`torch.bmm` 是 PyTorch 中的批量矩阵乘法函数，它对两个三维张量进行逐批次的矩阵乘法。具体规则如下：

- **输入张量形状**：
  - `input`: `(batch, n, m)`
  - `mat2`: `(batch, m, p)`
- **输出张量形状**：`(batch, n, p)`

其中 `batch` 是批次大小，必须相同；`n` 是第一个矩阵的行数；`m` 是第一个矩阵的列数和第二个矩阵的行数；`p` 是第二个矩阵的列数。

**运算过程**：对于批次中的每一个索引 `b`，执行普通矩阵乘法：`output[b] = input[b] @ mat2[b]`。所有批次的乘法并行计算，提高效率。

**要求**：
- 两个输入必须都是三维张量。
- 批次维度 `batch` 必须相等。
- 第一个张量的最后一个维度必须等于第二个张量的倒数第二个维度（即 `m`）。

### 3.3.4.2 torch.bmm 与 torch.matmul 的区别

|   特性   |               `torch.bmm`                |          `torch.matmul`          |
| :------: | :--------------------------------------: | :------------------------------: |
| 输入维度 |              仅支持三维张量              |  支持任意维度（≥1），并支持广播  |
| 广播机制 |                不支持广播                |    支持广播，可以自动扩展维度    |
|   性能   |      针对三维批量乘法优化，可能更快      |  通用性更强，但在某些情况下略慢  |
| 使用场景 | 明确的批量矩阵乘法，如注意力中的加权求和 | 通用矩阵乘法，适用于各种维度组合 |

**注意**：当两个输入都是三维且批次维度相同，且满足矩阵乘法维度要求时，两者结果一致。 

### 3.3.4.3 基本代码实现

#### 示例 1：基本用法

```python
import torch

batch_size = 2
input = torch.randn(batch_size, 2, 3)   # (2, 2, 3)
mat2 = torch.randn(batch_size, 3, 4)    # (2, 3, 4)

output = torch.bmm(input, mat2)         # (2, 2, 4)
print(output.shape)  # torch.Size([2, 2, 4])
```

#### 示例 2：在注意力机制中的应用

```python
# 模拟注意力权重和值
batch_size = 4
seq_len = 5
hidden_dim = 8

attn_weights = torch.randn(batch_size, seq_len, seq_len)  # (4, 5, 5)
values = torch.randn(batch_size, seq_len, hidden_dim)     # (4, 5, 8)

# 计算上下文向量：每个位置对值进行加权求和
context = torch.bmm(attn_weights, values)  # (4, 5, 8)
print(context.shape)  # torch.Size([4, 5, 8])
```

#### 示例 3：与 torch.matmul 对比

```python
output_bmm = torch.bmm(input, mat2)
output_matmul = torch.matmul(input, mat2)
print(torch.allclose(output_bmm, output_matmul))  # True
```

#### 注意事项

- 输入必须是三维张量；对于更高维度，需先重塑或使用 `torch.matmul`。
- 批次维度大小必须相同，否则会报错。
- 如果需要对不同批次使用不同形状的矩阵乘法，`torch.bmm` 不适用，可考虑 `torch.einsum` 或循环。



## 3.4 基本代码实现

进行一个Q查询 token 对 32 个值向量的注意力

对于编码器：前向传播算中调用RNN等模型，是一次就能把所有时间步算出来，因为输入张量中包含整个句子所有token的词向量，但不是同步算完，而是内部循环计算

代码实现如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 自定义注意力机制模块, 实现: Q(查询张量) 和 键值对(K, V)的注意力计算.
# 只是一个时间步的计算
class MyAttn(nn.Module):  
    # 1.1 初始化函数
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        """
        初始化函数, 用于初始化 注意力机制的核心参数.
        query_size: Q张量的维度
        key_size:   K张量的维度
        value_size1: V张量的序列长度(即: 原句子多少个单词)
        value_size2: V张量的维度(词向量), 例如: 64
        output_size: 输出张量的维度
        """
        # 1. 初始化父类的内容.
        super().__init__()

        # 2. 定义参数.
        self.query_size = query_size        # 查询张量Q的维度, 例如: 32
        self.key_size = key_size            # 键张量K的维度, 例如: 32
        self.value_size1 = value_size1      # 值张量V的序列长度, 例如: 32
        self.value_size2 = value_size2      # 值张量V的维度, 例如: 64
        self.output_size = output_size      # 输出张量的维度, 例如: 32

        # 3. 注意力权重计算层: 将Q和K拼接后, 映射到: 值序列长度维度.
        # 例如: 输入维度Q(32) + K(32) = 64,  输出维度: 32
        self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)

        # 4. 注意力融合层: 将原始Q 和 注意力加权后的V拼接(融合)后, 映射到: 输出维度.
        # 例如: 输入维度Q(32) + V(64) = 96,  输出维度: 32
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)


    # todo 1.2 前向传播, 计算: 注意力权重 和 输出.
    def forward(self, Q, K, V):
        """ 
        前向传播, 计算: 注意力权重 和 (最终)输出.
        Q: 查询张量, 形状: [1, 1, query_size] -> [1, 1, 32]
        K: 键张量, 形状: [1, 1, key_size] -> [1, 1, 32]
        V: 值张量, 形状: [1, value_size1, value_size2] -> [1, 32, 64]
        :return:
        """
        # 1. 计算注意力权重
        # 1.1 拼接Q和K, 维度变化: [1, 1, 32] + [1, 1, 32] = [1, 1, 64]
        # 对于q和k
        # 第一个1：batch 大小 
        # 第二个1：序列长度这里只有一个 token 
        # 32：查询向量的特征维度query_size，即32个单词
        # Q[0], K[0] 去掉第一个维度batch
        qk_cat = torch.cat((Q[0], K[0]), dim=-1) #[1, 32] + [1, 32] = [1, 64]
        print(f'qk_cat的形状: {qk_cat.shape}')

        # 1.2 通过线性层计算注意力得分, 维度变化: [1, 64] -> [1,  32]
        # 当前Q(一个token)对所有Key的相关性得分 这里的 32 对应的是 Key 的序列长度
        attn_scores = self.attn(qk_cat)
        print(f'attn_scores的形状: {attn_scores.shape}')

        # 1.3 用softmax()将得分 -> 概率分布:[1, 32] ->  [1, 32]
        attn_weights = F.softmax(attn_scores, dim=-1)
        print(f'attn_weights的形状: {attn_weights.shape}')

        # 2. 应用注意力权重(注意力分配系数) 到 值张量V
        # 2.1 扩展注意力权重维度, 以便匹配v的批次维度, 即: [1, 32] -> [1, 1, 32]
        attn_weights_expanded = attn_weights.unsqueeze(0)

        # 2.2 使用bmm()函数执行矩阵乘法, 维度变化: [1, 1, 32] * [1, 32, 64] = [1, 1, 64]
        # 这个操作相当于对 V 中每个位置(共32个) 按其注意力权重加权求和，得到一个 64 维的上下文向量c
        attn_applied = torch.bmm(attn_weights_expanded, V)
        print(f'attn_applied的形状: {attn_applied.shape}')

        # 3. 融合原始查询Q 和 注意力加权后的V
        # 3.1 拼接Q和V, 维度变化: [1, 1, 32] + [1, 1, 64] = [1, 1, 96]
        # 这里Q相当于 解码器的 隐藏状态 V代表的是上下文变量c 
        # 拼接 将两个向量直接连接成一个更长的向量，保留了各自原始的特征
        output_cat = torch.cat((Q, attn_applied), dim=-1)
        print(f'output_cat的形状: {output_cat.shape}')

        # 3.2 通过线性层降维到输出维度.
        # 维度变化: [1, 96] -> [1, 32] -> [1, 1, 32]
        output = self.attn_combine(output_cat)
        print(f'output的形状: {output.shape}')

        # 4. 返回结果
        return output, attn_weights         # output: [1, 1, 32], attn_weights: [1, 32]

if __name__ == '__main__':
    # 1. 实例化参数设置.
    query_size, key_size, value_size1, value_size2, output_size = 32, 32, 32, 64, 32

    # 2. 创建随机输入张量.
    Q = torch.randn(1, 1, query_size)               # Q  查询张量: [批次, 序列, 特征] -> [1, 1, 32]torch.Size([1, 32])
    K = torch.randn(1, 1, key_size)                 # K 键张量: [批次, 序列, 特征] -> [1, 1, 32]
    V = torch.randn(1, value_size1, value_size2)    # V 值张量: [批次, 单词数, 词向量维度] -> [1, 32, 64]

    # 3. 实例化自定义的注意力机制模块, 并测试.
    my_attn = MyAttn(query_size, key_size, value_size1, value_size2, output_size)
    output, attn_weights = my_attn(Q, K, V)

    # 4. 输出结果.
    print(' =.= ' * 40)
    print(f'查询张量Q的注意力结果表示: {output.shape}, {output}')               # torch.Size([1, 1, 32])
    print(f'查询张量Q的注意力权重分布: {attn_weights.shape}, {attn_weights}')   # torch.Size([1, 32])
```





## 3.5 Teacher Forcing

### 3.5.1 定义与概念

Teacher Forcing 是训练序列到序列（Seq2Seq）模型时常用的一种策略。在解码器逐步生成目标序列的过程中，每个时间步的输入**不使用模型自己上一步预测的输出**，而是使用**真实目标序列中的上一个词**（即 ground truth）作为当前时间步的输入。

这种训练方式类似于老师在每一步告诉学生正确答案，然后让学生根据正确答案继续往下预测，因此称为 Teacher Forcing。

### 3.5.2 为什么需要 Teacher Forcing

在 Seq2Seq 模型中，解码器是自回归的：生成当前词依赖于之前生成的词。如果训练时使用模型自己的预测作为下一步输入，会存在以下问题：

1. **训练初期预测质量差**：模型参数随机初始化时，预测的词很可能是错误的。一旦某个时间步预测错误，这个错误会传递到后续时间步，导致误差累积，模型很难从错误中恢复。
2. **收敛速度慢**：由于错误累积，模型需要很长时间才能学会生成合理的序列。
3. **训练不稳定**：模型可能陷入局部最优，难以学习到正确的序列模式。

Teacher Forcing 通过强制使用真实目标词作为下一步输入，**避免了误差累积，使模型能够更快、更稳定地收敛**。

### 3.5.3 工作原理与公式

假设目标序列为 $y = (y_1, y_2, \dots, y_T)$，解码器在时间步 $t$ 的隐藏状态为$ s_t$，输出为$ \hat{y}_t$

**1. 无 Teacher Forcing（自由运行）**

解码器每个时间步的输入是上一步的预测：

$$
\text{input}_t = \hat{y}_{t-1}
$$

$$
s_t = \text{Decoder}(s_{t-1}, \hat{y}_{t-1})
$$

$$
\hat{y}_t = \text{softmax}(W s_t + b)
$$

这种方式训练时容易误差累积，但推理时只能这样进行。

**2. 使用 Teacher Forcing**

解码器每个时间步的输入是真实目标词：

$$
\text{input}_t = y_{t-1}
$$

$$
s_t = \text{Decoder}(s_{t-1}, y_{t-1})
$$

$$
\hat{y}_t = \text{softmax}(W s_t + b)
$$

损失函数通常为交叉熵损失：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, \mathbf{x})
$$

其中 $P(y_t \mid y_{<t}, \mathbf{x})$ 是模型在时间步 t 预测真实目标词 $y_t$ 的概率。

**3. 训练与推理的不一致**

- **训练时**：使用 Teacher Forcing，输入是真实目标词。
- **推理时**：没有真实目标词，只能使用模型自己上一步的预测作为输入。

这种不一致称为 **曝光偏差（Exposure Bias）**。模型在训练时从未见过自己的错误预测，因此在推理时一旦预测错误，就可能产生连锁反应，导致生成质量下降。

### 3.5.4 代码实现

以下是在 PyTorch 中实现 Teacher Forcing 的简化示例。假设我们有一个解码器 `decoder`，它接收当前输入词索引和隐藏状态，返回输出和新的隐藏状态。

```python
import torch
import torch.nn as nn
import random

def train_step(encoder, decoder, src, trg, teacher_forcing_ratio=0.5):
    """
    src: 源序列，形状 [batch_size, src_len]
    trg: 目标序列，形状 [batch_size, trg_len]
    teacher_forcing_ratio: 使用真实目标词作为下一步输入的概率
    """
    batch_size = src.size(0)
    trg_len = trg.size(1)
    trg_vocab_size = decoder.output_size

    # 编码器前向
    encoder_outputs, hidden = encoder(src)

    # 解码器第一个输入通常是起始符 <sos>
    input_token = trg[:, 0]  # 形状 [batch_size]

    outputs = []

    for t in range(1, trg_len):
        # 解码器前向，注意输入需要增加序列长度维度
        output, hidden, attn_weights = decoder(input_token.unsqueeze(1), hidden, encoder_outputs)
        outputs.append(output)

        # 决定下一个输入是使用真实目标词还是模型预测词
        teacher_force = random.random() < teacher_forcing_ratio
        top1 = output.argmax(1)  # 模型预测的词，形状 [batch_size]
        input_token = trg[:, t] if teacher_force else top1

    outputs = torch.stack(outputs, dim=1)  # [batch_size, trg_len-1, trg_vocab_size]
    return outputs
```

**代码说明**：

- `teacher_forcing_ratio` 控制使用真实目标词的概率。通常训练初期设置较高（如 1.0），随着训练进行逐渐降低。
- `input_token` 是当前时间步的输入词索引，形状 `[batch_size]`，需要 `unsqueeze(1)` 变为 `[batch_size, 1]` 以匹配解码器输入。
- `top1 = output.argmax(1)` 获取模型预测概率最大的词。
- `trg[:, t]` 是真实目标序列中第 \(t\) 个词。

### 3.5.5 优缺点与改进

**优点**

- **加速收敛**：使用真实目标词作为输入，模型能快速学习正确的序列模式。
- **训练稳定**：避免误差累积，梯度传播更稳定。
- **简单有效**：实现容易，在大多数 Seq2Seq 任务中表现良好。

**缺点**

- **曝光偏差**：训练时使用真实目标，推理时使用模型预测，导致训练和推理不一致。
- **泛化能力受限**：模型可能过于依赖真实目标，无法很好地处理自身预测的错误。
- **不适合长序列**：对于很长的序列，Teacher Forcing 可能使模型缺乏纠正自身错误的能力。

### 3.5.6 改进方法：Scheduled Sampling

Scheduled Sampling 是一种逐渐从 Teacher Forcing 过渡到自由运行的方法。在训练初期使用较高的 Teacher Forcing 比例，随着训练进行逐渐降低比例，让模型逐渐适应自己的预测。

常用策略：

- **线性衰减**：$\text{teacher\_forcing\_ratio} = \max(0, 1 - \frac{\text{epoch}}{\text{total\_epochs}})$ 
- **指数衰减**：$\text{ratio} = k^{\text{epoch}} $，其中 \(0 < k < 1\)
- **随机采样**：以概率 $p$使用真实目标，$1-p$ 使用模型预测。





## 3.6 Seq2Seq案例——英译法案例

### 3.6.1 案例介绍

![image-20260908103533439](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260908103533439.png)

数据分析：

![image-20260908103806633](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260908103806633.png)



### 3.6.2 代码实现

#### 3.6.2.1 CUDA安装

打开cmd命令行，输入：

```bash
nvidia-smi
```

![image-20260908131114053](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260908131114053.png)

可以查看到CUDA版本

来到pytorch官网：https://pytorch.org/

选择合适的版本

![image-20260908131447982](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260908131447982.png)

复制下上图最下面的Run this Command

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

以管理员的身份运行Anacoda Prompt

创建一个新的沙箱： nlpbase_GPU:

```bash
conda create -n nlpbase_GPU python=3.10		#建议使用3.10版本，nlpbase_GPU就是沙箱名
```

切换到对应沙箱：

```bash
conda activate nlpbase_GPU					#切换沙箱
```

输出之前在官网粘贴的命令：

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

等待安装完成即可



#### 3.6.2.2 数据处理

对原始数据进行清洗和规范化处理，去除掉无用字符。

将原始数据-->tensor张量-->Dataset-->dataloader

代码如下：

```python
# 导包
import re               # 正则表达式相关
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm   # 进度条

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 设备选择, 我们可以选择在cuda 或者 cpu上运行你的代码.
# windows写法 如果有cuda, 就用cuda, 否则用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'当前设备: {device}')

# 指定特殊的token
# 起始标记
SOS_token = 0       # start of  sentence
# 结束标记
EOS_token = 1       # end of  sentence
# 最大句子长度不能超过10(包括标点)
MAX_LENGTH = 10
# 数据文件的路径
data_path = './data/eng-fra-v2.txt'

# 定义函数  对字符串规范化处理
def normalizeString(s):

    # 1. 将字符串转成小写形式, 并去除收尾空白字符.
    s = s.lower().strip()

    # 2. 在 .!? 前加1个空格, 使用正则表达式的捕获组替换.
    # 参1: 正则表达式(即: 要被替换的内容), 参2: 替换后的内容, 参3: 要操作的字符串
    s = re.sub('([.!?])', r' \1', s)

    # 3. 过滤非标准字符: 保留大小写字母 和 基本的标点符号, 其它字符替换为: 空格.
    # [^a-zA-Z.!?]解释:  除了大小写字母, .!? 符号之外, 任意的1个字符
    # +解释: 数量词, 代表前边的内容至少出现1次, 至多出现无数次.
    s = re.sub('[^a-zA-Z.!?]+', ' ', s)

    return s

# 数据预处理 -> 清洗文本 和 构建文本字典.
def my_getdata():
    # 1. 读取原始文件数据.
    # 1.1 打开文件, 使用 with open语法读取.
    with open(data_path, 'r', encoding='utf-8') as src_f:
        # 1.2 一次性读取所有行.
        lines = src_f.readlines()       # 格式为: ['第1行\n', '第2行\n', ...]
        # print(lines[:5])

        # 2. 清洗文本并构建双语 句子树(句子对)
        # for line in lines:         获取到每行数据,暂存到line
        # for s in line.split('\t'): 每行数据line按照\t切割,遇到\t就进行切割
        my_pairs = [ [normalizeString(s) for s in line.split('\t')] for line in lines]

        # 3. 初始化英语词汇表
        # 3.1 创建单词到索引的字典, 预定义特殊字符SOS(句子开始), EOS(句子结束)的索引为: 0, 1
        english_word2index = {'SOS': SOS_token, 'EOS': EOS_token}

        # 3.2 初始化英语词汇表大小计数器, 初始值为: 2, 因为已经包含: SOS和EOS
        english_word_n = 2

        # 4. 初始化法语词汇表 和 词汇表大小计数器
        french_word2index = {'SOS': SOS_token, 'EOS': EOS_token}
        french_word_n = 2

        # 5. (具体的)构建英语词汇表的动作.
        # 5.1 遍历所有的双语句子对, 获取英语句子中的单词.
        for pair in my_pairs:       # pair的数据格式: ['英语句子', '法语句子']
            # 5.2 对每个英语句子处理, 将句子按照 空格 切割成单词列表
            for word in pair[0].split(' '):
                # 5.3 检查单词是否在词汇表中, 不存在, 分配1个新索引, 新索引值 = 当前词汇表的大小
                if word not in english_word2index:
                    english_word2index[word] = english_word_n
                    english_word_n += 1

            # 6.4 构建法语词汇表
            for word in pair[1].split(' '):
                if word not in french_word2index:
                    french_word2index[word] = french_word_n
                    french_word_n += 1

        # 7. 构建反向映射表(索引到单词的映射)
        # 7.1 英语词汇表的反向映射.
        english_index2word = {v: k for k, v in english_word2index.items()}
        # 7.2 法语词汇表的反向映射.
        french_index2word = {v: k for k, v in french_word2index.items()}

        # 返回: 英语词汇表映射, 英语单词反向映射, 英语词汇表大小, 法语词汇表映射, 法语单词反向映射, 法语词汇表大小, 双语句子对.
        return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs

# 4. 数据预处理 -> 构建数据集对象(DataSet)
# 4.1 调用 my_getdata()函数, 获取: 预处理好的数据结构.
english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()

# 4.2 定义 MyPairsDataset类
class MyPairsDataset(Dataset):
    # 1. 初始化函数.
    def __init__(self, my_pairs):
        # 1.1 保存双语句子对.
        self.my_pairs = my_pairs
        # 1.2 计算样本总数, 样本数 = 双语句子对数.
        self.sample_len = len(my_pairs)

    # 2. 定义获取样本总数的方法.
    def __len__(self):
        return self.sample_len

    # 3. 定义获取单个样本的方法.
    def __getitem__(self, index):
        # 1. 修正索引值, 确保在有效范围内. 即: 索引不能小于0, 不能大于 样本总数 - 1
        index = min(max(index, 0), self.sample_len - 1)

        # 2. 按索引获取双语句子对, x: 英语句子, y: 法语句子.
        x = self.my_pairs[index][0]       
        y = self.my_pairs[index][1]  

        # 3. 英语句子文本转数值.
        # 3.1 按空格分割单词, 获取每个单词的索引.
        x = [english_word2index[word] for word in x.split(' ')]
        # 3.2 在句子末尾添加结束标记EOS
        x.append(EOS_token)
        # 3.3 将句子转换成张量, 并指定设备(CPU 或者 GPU)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        # 4. 法语句子文本转数值.
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        return tensor_x, tensor_y

# 5. 数据预处理 -> 定义函数, 获取DataLoader对象.
def get_dataloader():
    # 1. 实例化数据集对象.
    my_dataset = MyPairsDataset(my_pairs)

    # 2. 创建数据加载器对象.
    # 参1: 数据集对象.  参2: 批次大小.  参3: 是否打乱数据(训练集打乱, 测试集不打乱).
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
 
    # 7. 返回数据加载器对象.
    return my_dataloader

```



#### 3.6.2.3 模型构建

构建GRU编码器

![image-20260909090827896](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260909090827896.png)

构建GRU解码器

![image-20260909092558653](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260909092558653.png)



代码如下：

```python
# 6.构建基于GRU的编码器
"""
EncoderRNN类 实现思路分析:
    1. __init__函数, 定义 2 个层.
        self.embedding
        self.gru
    2. forward() 前向传播.
    3. 初始化 隐藏层输入数据. 
"""
class EncoderRNN(nn.Module):
    # 1. 初始化函数.
    def __init__(self, input_size, hidden_size):
        """
        初始化属性信息的.
        input_size: 编码器词嵌入层的输入维度, 即: 词汇表的大小(2803个英语单词)
        hidden_size: 编码器的隐藏层维度, 即: 隐藏层单元的个数(256个隐藏单元)
        """
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 保存输入参数.
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 3. 实例化词嵌入层.
        # 输入: [batch_size, seq_len],    输出: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(input_size, hidden_size)

        # 4. 实例化GRU层.
        # 参1: hidden_size: 输入的特征维度, 即: 词嵌入维度.
        # 参2: hidden_size: 隐藏层的维度, 即: 256
        # 参3: batch_first: 批次维数是否为第一个维度, 即(格式为): [批次大小, 句子长度, 词向量维度], 而不是默认的[句子长度, 批次大小, 词向量维度]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # 2. 前向传播函数.
    def forward(self, input, hidden):
        """
        前向传播函数.
        input:  输入的单词索引序列, 即: [batch_size, seq_len] -> [1, 6]
        hidden: 初始的隐藏状态, 即: [num_layer, batch_size, hidden_size] -> [1, 1, 256]
        :return:
        """
        # 1. 通过词嵌入层, 将单词索引序列 转换为 单词向量序列.
        # 输入形状: [batch_size, seq_len] -> [1, 6]
        # 输出形状: [batch_size, seq_len, hidden_size]  -> [1, 6, 256]
        output = self.embedding(input)

        # 2. GRU层处理.
        # 输入: batch_size不用对齐，但是数值要相同
        #   output: [batch_size, seq_len, hidden_size] >  [1, 6, 256]
        #   hidden: [num_layer, batch_size, hidden_size] > [1, 1, 256]
        # 输出:
        #   output: [batch_size, seq_len, hidden_size] > [1, 6, 256]
        #   hidden: [num_layer, batch_size, hidden_size] > [1, 1, 256]
        output, hidden = self.gru(output, hidden)

        # 3. 返回GRU层的输出和最终的隐藏状态.
        return output, hidden

    # 3.初始化 隐藏层输入数据.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 7. 测试基于GRU的编码器.
def test_encoder():
    # 1. 获取数据集加载器对象.
    my_dataloader = get_dataloader()

    # 2. 初始化编码器参数.
    vocab_size = english_word_n     # (英文)词汇表大小, 2803个英语单词
    hidden_size = 256               # 隐藏层维度, 256

    # 3. 创建编码器模型对象.
    my_encoder_gru = EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)

    # 4. (选做: CPU版可以不做) 将模型移动到指定的设备.
    my_encoder_gru = my_encoder_gru.to(device)

    # 5. 遍历数据集加载器, 进行测试.
    for i, (x, y) in enumerate(my_dataloader):
        # 5.1 打印输入英文句子索引张量 和 输入张量的形状.
        print(f'x: {x.shape}, {x}')     # x: torch.Size([1, 6]), tensor([14, 15, 43, 139, 888, 1])

        # 5.2 获取编码器的初始隐藏状态.
        h0 = my_encoder_gru.init_hidden()

        # 5.3 执行前向传播.
        output, hn = my_encoder_gru(x, h0)

        # 5.4 打印输出结果形状.
        print(f'output: {output.shape}')    # torch.Size([1, 6, 256])

        # 5.5 打印隐藏状态的形状.
        print(f'hidden: {hn.shape}')        # torch.Size([1, 1, 256])
        break       # 只看一组, 实际开发, 千万不要写.

# 8. 构建基于GRU的解码器 -> 版本1: 无Attention(注意力机制)
class DecoderRNN(nn.Module):
    # 1. 初始化函数.
    def __init__(self, output_size, hidden_size):
        """
        初始化属性信息
        output_size: 解码器输出维度, 即: 目标语言(法语)词汇表大小
        hidden_size: 解码器隐藏层维度, 即: 每个词向量的特征数(256)
        """
        # 1. 初始化父类信息
        super().__init__()
        # 2. 保存输入参数.
        self.output_size = output_size
        self.hidden_size = hidden_size
        # 3. 创建词嵌入层, 输入: [batch_size, seq_len], 输出: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 4. 创建GRU层.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 5. 创建线性层(输出层)
        self.out = nn.Linear(hidden_size, output_size)
        # 6(了解). 创建softmax层.
        self.softmax = nn.LogSoftmax(dim=-1)

    # 2. 前向传播函数.
    def forward(self, input, hidden):
        # 1. 词嵌入处理.
        output = self.embedding(input)
        # 2. ReLU激活函数处理.
        output = F.relu(output)
        # 3. GRU层处理 -> 大白话处理: 都是[1, 1, 256]
        # 输入时: output -> [batch_size, seq_len, hidden_size],   hidden -> [num_layers, batch_size, hidden_size]
        # 输出时: output -> 形状同上,   hidden -> 形状同上
        output, hidden = self.gru(output, hidden)

        # 4. 线性层 和 softmax层处理.
        output = self.softmax(self.out(output[0]))

        return output, hidden

    # 3. 初始化隐藏层输入数据.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 9. 测试基于GRU的解码器 -> 测试版本1: 无Attention(注意力机制)
def test_decoder(): 
    # 1. 获取数据集加载器对象.
    my_dataloader = get_dataloader()
    # 2. 初始化编码器模型, 并移动到GPU.
    my_encoder_gru = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device)
    print(f'my_encoder_gru: {my_encoder_gru}')
    
    # 3. 初始化解码器模型, 并移动到GPU.
    my_decoder_gru = DecoderRNN(output_size=french_word_n, hidden_size=256).to(device)
    print(f'my_decoder_gru: {my_decoder_gru}')

    # 4. 完整的编码 -> 解码流程测试.
    # 4.1 从数据集加载器中获取1个批次的样本(即: 1条数据)
    for i, (x, y) in enumerate(my_dataloader):
        # 4.2 打印输入数据的信息.
        # print(f'输入数据信息(英语句子): {x.shape}, {x}')  # 输入数据信息(英语句子): torch.Size([1, 6]), tensor([[ 77,  78, 147,  24,   4,   1]])
        # print(f'输入数据信息(法语句子): {y.shape}, {y}')  # 输入数据信息(法语句子): torch.Size([1, 7]), tensor([[123, 297, 126, 246, 384,   5,   1]])
        print(f'输入数据信息(英语句子): {x.shape}')
        print(f'输入数据信息(法语句子): {y.shape}')

        # 4.3 编码过程: 将英文句子编码为 隐藏状态.
        # 4.3.1 初始化编码器.
        h0 = my_encoder_gru.init_hidden()
        # 4.3.2 (编码器)前向传播.
        encoder_output_c, hidden = my_encoder_gru(x, h0)
        

        # 4.4 解码过程: 将隐藏状态解码为法语句子.
        # print(f'观察: 最后1个时间步的output输出: {encoder_output_c[0][-1].shape}') # [6, 256] -> [256]

        # 4.4.1 具体的解码过程 -> 逐个字符生成, 将隐藏状态解码为法语句子.
        # 遍历目标句子(法语句子)的每个时间步.
        for i in range(y.shape[1]):
            # 4.4.2 提取当前时间步的 目标词索引.
            # y[0][i]:     取出batch中第1个样本的第i个词的索引.
            # view(1, -1): 将标量转为[1, 1]的形状, 匹配: 解码器输入要求.
            tmp = y[0][i].view(1, -1)
            # 4.4.3 解码器的前向传播.
            output, hidden = my_decoder_gru(tmp, hidden)
            # 4.4.4 打印解码器的输出信息.
            print(f'第 {i + 1} 个法语单词的预测概率分布: {output.shape}, {output.shape}')

        print('\n' * 5)

# 10. 构建基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制) 
class AttnDecoderRNN(nn.Module):
    # todo 10.1 初始化函数.
    # 参1: 目标(法语)词汇表大小, 参2: 隐藏层维度(和编码器一致), 参3: 随机失活概率, 参4: (句子)最大长度
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 保存输入参数
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 3. 创建词嵌入层.
        # 输入形状: [batch_size, seq_len] -> [1, 1]
        # 输出形状: [batch_size, seq_len, hidden_size] -> [1, 1, 256]
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 4. 注意力权重计算层: 计算查询向量 和 编码器输出的匹配程度.
        # 参1: 拼接后的查询向量 和 隐藏状态 -> [1, 1, 512]
        # 参2: 注意力的权重分布 -> [1, 1, 10]  (最大)10个词
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # 5. 注意力融合层, 将 词嵌入 和 注意力权重进行融合.  
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 6. dropout层, 随机丢弃部分神经元, 防止过拟合.
        self.dropout = nn.Dropout(self.dropout_p)

        # 7. 创建GRU层: 处理序列数据, 维持隐藏状态.
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 8. 输出层: 将GRU的隐藏状态映射为: (法语)目标词汇表大小.
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # 9. 对数softmax()层, 将输出映射为 对数softmax()概率分布
        self.softmax = nn.LogSoftmax(dim=-1)


    # todo 10.2 前向传播函数.
    # 参1: input 当前时间步的输入词索引 -> [batch_size, 1]
    # 参2: hidden 上一个时间步的隐藏状态 -> [1, batch_size, hidden_size]
    # 参3: encoder_outputs 编码器所有时间步的输出 -> [batch_size, seq_len, hidden_size]
    def forward(self, input, hidden, encoder_outputs):
        # 1. 词嵌入层, 输入形状: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        # [1, 1] -> [1, 1, 256]
        embedded = self.embedding(input)

        # 2. 应用Dropout(随机失活层), 随机失活.
        embedded = self.dropout(embedded)

        # 3. 计算注意力权重.
        # step1: torch.cat((embedded[0], hidden[0]), 1))                ->  [1, 512]
        # step2: self.attn(torch.cat((embedded[0], hidden[0]), 1)       ->  [1, 10]
        # step3: 应用softmax()层, 映射为: [1, 10] -> [1, 10]
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 4. 计算注意力上下文.
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # 5.注意力融合层.
        output = torch.cat((embedded[0], attn_applied[0]), 1)       # [1, 1, 512]
        output = self.attn_combine(output).unsqueeze(0)                          # [1, 1, 256]
        # 6. 激活函数处理.
        output = F.relu(output)
        # 7. GRU层: 处理序列数据, 维持隐藏状态.
        output, hidden = self.gru(output, hidden)       # [1, 1, 256]
        # 8. 输出层: 将GRU的隐藏状态映射为: (法语)目标词汇表大小.
        output = self.softmax(self.out(output[0]))      # [1, 4345]

        # 9. 返回结果.
        # 参1: 当前时间步的输出概率分布          ->  [1, 4345]
        # 参2: 更新后的隐藏状态(本次的隐藏状态)   ->  [1, 1, 256]
        # 参3: 注意力权重分布(用于可视化分析, 如果不做, 可以不返回)     -> [1, 10]
        return output, hidden, attn_weights


    # todo 10.3 定义初始化的隐藏状态.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# todo 11. 测试基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制)
def test_attn_decoder():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()

    # 2. 模型初始化阶段
    # 2.1 创建编码器对象.
    # 参1: 英语词汇表大小(2803), 参2: 隐藏层维度
    my_encoder = EncoderRNN(english_word_n, 256).to(device)

    # 2.2 创建解码器对象.
    # 参1: 法语词汇表大小(4345), 参2: 隐藏层维度
    my_decoder = AttnDecoderRNN(french_word_n, 256).to(device)

    # 3. 模型训练(推理)阶段
    # 3.1 从数据加载器中获取1个样本.
    for i, (x, y) in enumerate(my_dataloader):
        # 3.2 打印输入信息
        print(f'x(英语句子): {x.shape}, {x}')       # torch.Size([1, 6]),
        print(f'y(法语句子): {y.shape}, {y}')       # torch.Size([1, 8]),

        # 3.3 编码过程 -> 将英语句子 编码成 隐藏状态 -> (中间语义张量C)
        hidden = my_encoder.init_hidden()
        # output: 所有时间步的隐藏状态: [1, seq_len, 256]
        # hidden: 最后一个时间步的隐藏状态: [1, 1, 256]
        output, hidden = my_encoder(x, hidden)

        # 3.4 准备编码器的输出 -> 用于Attention机制.
        # 创建1个固定大小的张量, 用于存储编码器输出, 形状为: [10, 256]
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)

        # 3.5 将编码器实际输出 复制到 固定大小的张量中.
        for idx in range(output.shape[1]):
             encoder_output_c[idx] = output[0, idx]

        # 3.6 解码过程: 将隐藏状态 解码为 法语句子(必须逐词翻译)
        # 3.6.1 遍历目标句子的每个时间步.
        for i in range(y.shape[1]):
            # 3.6.2 提取当前时间步的目标词索引.
            tmp = y[0][i].view(1, -1)       # 例如: [[1595]]

            # 3.6.3 执行解码器的前向传播.
            # (实际)参数列表解释
            # tmp: 当前时间步的输入词索引, 形状: [1, 1]
            # hidden: 上一个时间步的隐藏状态, 形状: [1, 1, 256]
            # encoder_output_c: 编码器(所有时间步)的输出, 形状: [10, 256]

            # 返回值参数列表解释:
            # output: 当前时间步的输出概率分布, 形状: [1, 4345]
            # hidden: 当前时间步的隐藏状态, 形状: [1, 1, 256]
            # attn_weights: 当前时间步的注意力权重分布, 形状: [1, 10]

            #                                          Q    K         V
            output, hidden, attn_weights = my_decoder(tmp, hidden, encoder_output_c)

            # 3.6.4 打印结果.
            print(f'解码output.shape: {output.shape}')    # [1, 4345]
            print(f'解码hidden.shape: {hidden.shape}')    # [1, 1, 256]
            print(f'解码attn_weights.shape: {attn_weights.shape}')    # [1, 10]
            print('\n' * 3)

        # 只看1个句子(样本)
        break

```



#### 3.6.2.4 模型训练

代码如下：

```python
# 12. 构建模型内部迭代训练函数 -> 即: 完成单批次的训练过程
# 12.1 定义模型训练参数
# 学习率, 训练轮数, Teacher_Forcing比例, 输出信息打印间隔(每训练1000条打印一次),  绘图间隔(每训练100条绘图一次)
my_lr, epochs, teacher_forcing_ratio, print_interval_num, plot_interval_num = 1e-4, 1, 0.5, 1000, 100

# 12.2 定义函数, 实现: 单批次训练, 完成1个样本的 编码 -> 解码 -> 反向传播 -> 优化参数...
def train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss):
    """
    函数作用, 实现: 单批次训练, 完成1个样本的 编码 -> 解码 -> 反向传播 -> 优化参数...
    x: 输入序列, 即: 英语句子, 形状为: [batch_size = 1, seq_len]
    y: 目标序列, 即: 法语句子, 形状为: [batch_size = 1, seq_len]
    my_encoder_rnn: 编码器对象
    my_attn_decoder_rnn: 解码器对象(带注意力机制)
    myadam_encode: 编码器优化器
    myadam_decode: 解码器优化器
    my_crossentropy_loss: 损失函数
    """
    # 1. 编码阶段, 将输入的序列转换为上下文向量, 初始的隐藏状态: [1, 1, 256]
    encoder_hidden = my_encoder_rnn.init_hidden()

    # 编码器的前向传播.
    encoder_output, encoder_hidden = my_encoder_rnn(x, encoder_hidden)

    # 2. 解码参数准备
    # 2.1 构建编码器的输出张量, 用于: 注意力计算, 形状为: [10, 256]
    encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
    # 2.2 复刻实际的编码器输出 -> 固定长度张量, 即: [假设6个单词, 256] -> [10, 256]
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]

    # 2.3 解码器初始化隐藏状态.
    decoder_hidden = encoder_hidden     # [1, 1, 256]

    # 2.4 解码器的初始输入.
    input_y = torch.tensor([[SOS_token]], device=device)        # [1, 1]

    # 3. 初始化损失值
    my_loss = 0.0
    y_len = y.shape[1]      # 目标序列长度(要预测的法语句子长度), 例如: 9

    # 4. 根据概率值决定是否用 Teacher Forcing.
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # 4.1 走这里, 说明使用 Teacher Forcing, 就: 用真实标签作为下一步的输入.
        for i in range(y_len):
            # 4.1.1 解码器的前向传播
            # 输入: input_y -> [1, 1], decoder_hidden -> [1, 1, 256], encoder_output_c -> [10, 256]
            # 输出: output_y -> [1, 4345], decoder_hidden -> [1, 1, 256], attn_weights -> [1, 10]
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # 4.1.2 获取当前时间步的真实标签.
            target_y = y[0][i].view(1)
            # 4.1.3 累加损失
            my_loss += my_crossentropy_loss(output_y, target_y)
            # 4.1.4 下个时间步的输入 直接使用 真实标签.
            input_y = y[0][i].view(1, -1)   # [1, 1]
    else:
        # 4.2 非Teacher Forcing, 就: 用上一时间步的预测结果作为下一步的输入.
        for i in range(y_len):
            # 4.2.1 解码器前向传播
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # 4.2.2 获取当前时间步的真实标签.
            target_y = y[0][i].view(1)
            # 4.2.3 累加损失
            my_loss += my_crossentropy_loss(output_y, target_y)
            # 4.2.4 获取预测的下一个词, 即: 获取概率最高的词的索引和概率.
            topv, topi = output_y.topk(1)
            # 4.2.5 如果预测到句子的结束标记, 则停止预测.
            if topi.squeeze().item() == EOS_token:
                break
            # 4.2.6 走到这里, 说明没有预测到结束标记, 则: 将预测的词 作为 下一步的输入.
            input_y = topi.detach()     # [1, 1]

    # 5. 反向传播和参数更新.
    myadam_encode.zero_grad()       # 梯度清零, 编码器.
    myadam_decode.zero_grad()       # 梯度清零, 解码器.

    my_loss.backward()              # 反向传播.
    myadam_encode.step()            # 参数更新, 编码器.
    myadam_decode.step()            # 参数更新, 解码器.

    # 6. 返回平均损失.
    return my_loss.item() / y_len

# 13. 构建模型训练函数 -> 即: 完成所有批次的训练过程, 即: 多轮, 多批次训练过程.
def train_seq2seq():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()
    # 2. 模型初始化, 记得把模型移动到GPU上, 我的电脑(训练1轮): CPU训练时间 50分钟, GPU训练时间: 20分钟
    # 2.1 编码器, 输入维度 = 英文词汇表大小2803, 隐藏层维度: 256
    my_encoder_rnn = EncoderRNN(english_word_n, 256).to(device)
    # 2.2 解码器, 输入维度 = 法语词汇表大小4345, 隐藏层维度: 256
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256, 0.1, 10).to(device)

    # 3. 优化器初始化, 使用: Adam优化器, 学习率: 1e-4
    myadam_encode = optim.Adam(my_encoder_rnn.parameters(), lr=my_lr)       # 编码器优化器
    myadam_decode = optim.Adam(my_attn_decoder_rnn.parameters(), lr=my_lr)  # 解码器优化器

    # 4. 损失函数初始化, 使用: NLLLoss
    my_crossentropy_loss = nn.NLLLoss()

    # 5. 训练参数初始化.
    plot_loss_list = []         # 存储绘图用的损失值.

    # 6. 具体的多轮, 多批次训练过程.
    # 6.1 外层循环, 控制训练轮数.
    for epoch_idx in range(1, epochs + 1):
        # 6.1.1 初始化本轮的损失累加器
        print_loss_total, plot_loss_total = 0.0, 0.0
        # 6.1.2 记录本轮开始训练时间.
        start_time = time.time()

        # 6.2 内层循环, 遍历数据集的每个样本(即: 每轮具体的 所有批次训练过程)
        for item, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            # 6.2.1 调用内部训练函数, 完成: 单批次(单样本)的训练过程.
            myloss = train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss)

            # 6.2.2 累加损失.
            print_loss_total += myloss
            plot_loss_total += myloss

            # 6.3 打印训练日志(每 print_interval_num=1000 个样本打印一次)
            if item % print_interval_num == 0:
                # 计算平均损失.
                print_loss_avg = print_loss_total / print_interval_num
                # 重置损失累加器.
                print_loss_total = 0.0
                # 打印训练信息: 轮次, 平均损失, 耗时.
                print(f'轮次: {epoch_idx}, 平均损失: {print_loss_avg:.4f}, 耗时: {time.time() - start_time:.4f} s(秒)!')

            # 6.4 记录损失用于绘图(每 plot_interval_num=100 个样本记录一次)
            if item % plot_interval_num == 0:
                # 计算平均损失.
                plot_loss_avg = plot_loss_total / plot_interval_num
                # 存储损失值.
                plot_loss_list.append(plot_loss_avg)
                # 重置损失累加器.
                plot_loss_total = 0.0

            # 扩展: 每轮训练3000个样本后, 结束训练, 实际开发, 这个代码万万不能写.
            # if item > 3000:
            #    break

        # 6.3 走到这里, 说明一轮训练完毕, 保存模型.
        torch.save(my_encoder_rnn.state_dict(), f'./model/my_encoder_rnn_{epoch_idx}.pth')  # pickle文件后缀, .pth, .pkl, .pickle
        torch.save(my_attn_decoder_rnn.state_dict(), f'./model/my_attn_decoder_rnn_{epoch_idx}.pth')

    # 7. 训练结束后, 绘制损失曲线.
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/Seq2Seq_loss.png')
    plt.show()

    # 8. 训练结束, 返回结果.
    return plot_loss_list           # 训练损失列表(每训练100个样本的平均损失)

```



#### 3.6.2.5 模型预测

代码如下：

```python
# todo 14. 构建模型评估 -> 用训练好的seq2seq模型进行 翻译.
def evaluate_seq2seq(x, my_encoder_rnn, my_attn_decoder_rnn):   # 英语句子, 编码器, 解码器(带注意力机制)
    # 0. 关闭梯度计算, 节省内存并加速推理 -> 只适用于 模型的预测过程.
    with torch.no_grad():
        # 1.编码阶段: 将输入的英文句子 -> 隐藏状态.
        encode_hidden = my_encoder_rnn.init_hidden()
        # 本次的输出, 本次的隐藏状态 = 编码器模型(本次的输入, 上一时刻的隐藏状态)  
        encode_output, encode_hidden = my_encoder_rnn(x, encode_hidden)

        # 2. 解码器参数准备.
        # 2.1 构建固定长度的编码器输出张量.
        encode_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
        for idx in range(x.shape[1]):
            encode_output_c[idx] = encode_output[0, idx]

        # 2.2 解码器的隐藏状态. 
        decode_hidden = encode_hidden

        # 2.3 解码器的初始输入, 句子的开始标记.
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3. 自回归解码过程(逐个生成目标句子)
        # 定义遍历, 记录: 存储解码后的法语单词
        decode_words = []
        # 初始化注意力矩阵
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        # 遍历, 开始解码.
        for idx in range(MAX_LENGTH):
            # 3.1 解码器, 前向传播.
            # 输入: 当前输入词的索引(Q), 解码隐藏状态(K), 编码器输出张量(V))
            # 输出: 下个词的概率分布, 更新后的隐藏状态, 注意力权重分布矩阵
            output_y, decode_hidden, attn_weights = my_attn_decoder_rnn(input_y, decode_hidden, encode_output_c)

            # 3.2 记录注意力矩阵.
            decoder_attentions[idx] = attn_weights
            # 3.3 预测下个词.
            topv, topi = output_y.topk(1)
            # 3.4 处理终止条件, 如果预测到EOS标记, 结束生成.
            if topi.squeeze().item() == EOS_token:
                break
            else:
                # 3.5 否则, 则添加预测的词到结果列表.
                decode_words.append(french_index2word[topi.squeeze().item()])
            # 3.6 更新输入: 把当前预测词作为下个时间步的输入.
            input_y = topi.detach()

        # 4. 返回解码结果 和 注意力矩阵.
        return decode_words, decoder_attentions[:idx + 1]


# todo 15. 模型评估函数调用, 记载模型, 对比 自定义样本进行 翻译.
# 模型路径
PATH1 = './model/my_encoder_rnn_1.pth'
PATH2 = './model/my_attn_decoder_rnn_1.pth'

# 定义函数
def test_seq2seq_evaluate():
    # 1. 获取数据加载器对象
    my_dataloader = get_dataloader()
    # 2. 加载编码器模型.
    my_encoder_rnn = EncoderRNN(english_word_n, hidden_size=256).to(device)

    # map_location: 确保在CPU和GPU(Cuda)都能加载. 正常应该是用GPU训练, 就用GPU预测, 用CPU训练, 就用CPU预测
    # 写了 map_location能实现: 用GPU训练, 用CPU预测
    # weights_only: 只加载模型权重参数
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), False)
    print(f'my_encoder_rnn编码器模型架构: {my_encoder_rnn}')

    # 3. 加载解码器模型
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), False)
    print(f'my_attn_decoder_rnn解码器模型架构: {my_attn_decoder_rnn}')

    # 4. 自定义测试样本.
    my_sample_pairs = [
        # 格式: ['英文句子', '法语句子']
        ['i m the spokesperson for this organization .', 'je suis le porte parole de cette institution .'],
        ['i am thinking about buying a new parasol .', 'je songe a acheter un nouveau parasol .'],
        ['he is able to swim very fast .', 'il est capable de nager tres vite .']
    ]
    print(f'自定义测试样本: {my_sample_pairs}')

    # 5. 对每个样本进行翻译.
    for index, pair in enumerate(my_sample_pairs):
        x = pair[0]     # 英语句子
        y = pair[1]     # 法语句子

        # 5.1 文本数值化, 英语句子 -> 索引序列
        tmpx = [english_word2index[word] for word in x.split(' ')]
        tmpx.append(EOS_token)      # 添加结束标记.
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 5.2 模型预测.
        decode_words, attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)

        # 5.3 把翻译结果转成句子格式.
        output_sentence = ' '.join(decode_words)
        print(f'输入(原始英文句子): {x}')
        print(f'输入(原始法语句子): {y}')
        print(f'输出(翻译后的法语句子): {output_sentence}')
        print(' -.- ' * 10, '\n')

```

效果图：颜色越浅，依赖越深

横坐标：法语句子	纵坐标：英语句子

![image-20260911081717454](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911081717454.png)

#### 3.6.2.6 完整代码

代码如下：

```python
"""
基于GRU的seq2seq模型架构实现翻译的过程:
    step1: 导入工具包 和 工具函数
    step2: 对持久化文件中数据进行预处理, 以满足模型训练要求
    step3: 构建基于GRU的编码器和解码器
    step4: 构建模型训练函数, 并进行训练
    step5: 构建模型评估函数, 并进行测试以及Attention效果分析.
"""

# 导包
import re               # 正则表达式相关
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm   # 进度条

#VS Code 终端默认编码不是 UTF-8 不是VS Code 可以不用写
import sys
sys.stdout.reconfigure(encoding='utf-8') # 强制让控制台使用 UTF-8 编码输出文字

# 设备选择, 我们可以选择在cuda 或者 cpu上运行你的代码.
# windows写法 如果有cuda, 就用cuda, 否则用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'当前设备: {device}')

# 指定特殊的token
# 起始标记
SOS_token = 0       # start of  sentence
# 结束标记
EOS_token = 1       # end of  sentence
# 最大句子长度不能超过10(包括标点)
MAX_LENGTH = 10
# 数据文件的路径
data_path = './data/eng-fra-v2.txt'

# 定义函数  对字符串规范化处理
def normalizeString(s):

    # 1. 将字符串转成小写形式, 并去除收尾空白字符.
    s = s.lower().strip()

    # 2. 在 .!? 前加1个空格, 使用正则表达式的捕获组替换.
    # 参1: 正则表达式(即: 要被替换的内容), 参2: 替换后的内容, 参3: 要操作的字符串
    s = re.sub('([.!?])', r' \1', s)

    # 3. 过滤非标准字符: 保留大小写字母 和 基本的标点符号, 其它字符替换为: 空格.
    # [^a-zA-Z.!?]解释:  除了大小写字母, .!? 符号之外, 任意的1个字符
    # +解释: 数量词, 代表前边的内容至少出现1次, 至多出现无数次.
    s = re.sub('[^a-zA-Z.!?]+', ' ', s)

    return s

# 数据预处理 -> 清洗文本 和 构建文本字典.
def my_getdata():
    # 1. 读取原始文件数据.
    # 1.1 打开文件, 使用 with open语法读取.
    with open(data_path, 'r', encoding='utf-8') as src_f:
        # 1.2 一次性读取所有行.
        lines = src_f.readlines()       # 格式为: ['第1行\n', '第2行\n', ...]

        # 2. 清洗文本并构建双语 句子树(句子对)
        # for line in lines:         获取到每行数据,暂存到line
        # for s in line.split('\t'): 每行数据line按照\t切割,遇到\t就进行切割
        my_pairs = [ [normalizeString(s) for s in line.split('\t')] for line in lines]

        # 3. 初始化英语词汇表
        # 3.1 创建单词到索引的字典, 预定义特殊字符SOS(句子开始), EOS(句子结束)的索引为: 0, 1
        english_word2index = {'SOS': SOS_token, 'EOS': EOS_token}

        # 3.2 初始化英语词汇表大小计数器, 初始值为: 2, 因为已经包含: SOS和EOS
        english_word_n = 2

        # 4. 初始化法语词汇表 和 词汇表大小计数器
        french_word2index = {'SOS': SOS_token, 'EOS': EOS_token}
        french_word_n = 2

        # 5. (具体的)构建英语词汇表的动作.
        # 5.1 遍历所有的双语句子对, 获取英语句子中的单词.
        for pair in my_pairs:       # pair的数据格式: ['英语句子', '法语句子']
            # 5.2 对每个英语句子处理, 将句子按照 空格 切割成单词列表
            for word in pair[0].split(' '):
                # 5.3 检查单词是否在词汇表中, 不存在, 分配1个新索引, 新索引值 = 当前词汇表的大小
                if word not in english_word2index:
                    english_word2index[word] = english_word_n
                    english_word_n += 1

            # 6.4 构建法语词汇表
            for word in pair[1].split(' '):
                if word not in french_word2index:
                    french_word2index[word] = french_word_n
                    french_word_n += 1

        # 7. 构建反向映射表(索引到单词的映射)
        # 7.1 英语词汇表的反向映射.
        english_index2word = {v: k for k, v in english_word2index.items()}
        # 7.2 法语词汇表的反向映射.
        french_index2word = {v: k for k, v in french_word2index.items()}

        # 返回: 英语词汇表映射, 英语单词反向映射, 英语词汇表大小, 法语词汇表映射, 法语单词反向映射, 法语词汇表大小, 双语句子对.
        return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs

# 4. 数据预处理 -> 构建数据集对象(DataSet)
# 4.1 调用 my_getdata()函数, 获取: 预处理好的数据结构.
english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()

# 4.2 定义 MyPairsDataset类
class MyPairsDataset(Dataset):
    # 1. 初始化函数.
    def __init__(self, my_pairs):
        # 1.1 保存双语句子对.
        self.my_pairs = my_pairs
        # 1.2 计算样本总数, 样本数 = 双语句子对数.
        self.sample_len = len(my_pairs)

    # 2. 定义获取样本总数的方法.
    def __len__(self):
        return self.sample_len

    # 3. 定义获取单个样本的方法.
    def __getitem__(self, index):
        # 1. 修正索引值, 确保在有效范围内. 即: 索引不能小于0, 不能大于 样本总数 - 1
        index = min(max(index, 0), self.sample_len - 1)

        # 2. 按索引获取双语句子对, x: 英语句子, y: 法语句子.
        x = self.my_pairs[index][0]       
        y = self.my_pairs[index][1]  

        # 3. 英语句子文本转数值.
        # 3.1 按空格分割单词, 获取每个单词的索引.
        x = [english_word2index[word] for word in x.split(' ')]
        # 3.2 在句子末尾添加结束标记EOS
        x.append(EOS_token)
        # 3.3 将句子转换成张量, 并指定设备(CPU 或者 GPU)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        # 4. 法语句子文本转数值.
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        return tensor_x, tensor_y

# 5. 数据预处理 -> 定义函数, 获取DataLoader对象.
def get_dataloader():
    # 1. 实例化数据集对象.
    my_dataset = MyPairsDataset(my_pairs)

    # 2. 创建数据加载器对象.
    # 参1: 数据集对象.  参2: 批次大小.  参3: 是否打乱数据(训练集打乱, 测试集不打乱).
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)
 
    # 7. 返回数据加载器对象.
    return my_dataloader

# 6.构建基于GRU的编码器
"""
EncoderRNN类 实现思路分析:
    1. __init__函数, 定义 2 个层.
        self.embedding
        self.gru
    2. forward() 前向传播.
    3. 初始化 隐藏层输入数据. 
"""
class EncoderRNN(nn.Module):
    # 1. 初始化函数.
    def __init__(self, input_size, hidden_size):
        """
        初始化属性信息的.
        input_size: 编码器词嵌入层的输入维度, 即: 词汇表的大小(2803个英语单词)
        hidden_size: 编码器的隐藏层维度, 即: 隐藏层单元的个数(256个隐藏单元)
        """
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 保存输入参数.
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 3. 实例化词嵌入层.
        # 输入: [batch_size, seq_len],    输出: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(input_size, hidden_size)

        # 4. 实例化GRU层.
        # 参1: hidden_size: 输入的特征维度, 即: 词嵌入维度.
        # 参2: hidden_size: 隐藏层的维度, 即: 256
        # 参3: batch_first: 批次维数是否为第一个维度, 即(格式为): [批次大小, 句子长度, 词向量维度], 而不是默认的[句子长度, 批次大小, 词向量维度]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # 2. 前向传播函数.
    def forward(self, input, hidden):
        """
        前向传播函数.
        input:  输入的单词索引序列, 即: [batch_size, seq_len] -> [1, 6]
        hidden: 初始的隐藏状态, 即: [num_layer, batch_size, hidden_size] -> [1, 1, 256]
        :return:
        """
        # 1. 通过词嵌入层, 将单词索引序列 转换为 单词向量序列.
        # 输入形状: [batch_size, seq_len] -> [1, 6]
        # 输出形状: [batch_size, seq_len, hidden_size]  -> [1, 6, 256]
        output = self.embedding(input)

        # 2. GRU层处理.
        # 输入: batch_size不用对齐，但是数值要相同
        #   output: [batch_size, seq_len, hidden_size] >  [1, 6, 256]
        #   hidden: [num_layer, batch_size, hidden_size] > [1, 1, 256]
        # 输出:
        #   output: [batch_size, seq_len, hidden_size] > [1, 6, 256]
        #   hidden: [num_layer, batch_size, hidden_size] > [1, 1, 256]
        output, hidden = self.gru(output, hidden)

        # 3. 返回GRU层的输出和最终的隐藏状态.
        return output, hidden

    # 3.初始化 隐藏层输入数据.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 7. 测试基于GRU的编码器.
def test_encoder():
    # 1. 获取数据集加载器对象.
    my_dataloader = get_dataloader()

    # 2. 初始化编码器参数.
    vocab_size = english_word_n     # (英文)词汇表大小, 2803个英语单词
    hidden_size = 256               # 隐藏层维度, 256

    # 3. 创建编码器模型对象.
    my_encoder_gru = EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)

    # 4. (选做: CPU版可以不做) 将模型移动到指定的设备.
    my_encoder_gru = my_encoder_gru.to(device)

    # 5. 遍历数据集加载器, 进行测试.
    for i, (x, y) in enumerate(my_dataloader):
        # 5.1 打印输入英文句子索引张量 和 输入张量的形状.
        print(f'x: {x.shape}, {x}')     # x: torch.Size([1, 6]), tensor([14, 15, 43, 139, 888, 1])

        # 5.2 获取编码器的初始隐藏状态.
        h0 = my_encoder_gru.init_hidden()

        # 5.3 执行前向传播.
        output, hn = my_encoder_gru(x, h0)

        # 5.4 打印输出结果形状.
        print(f'output: {output.shape}')    # torch.Size([1, 6, 256])

        # 5.5 打印隐藏状态的形状.
        print(f'hidden: {hn.shape}')        # torch.Size([1, 1, 256])
        break       # 只看一组, 实际开发, 千万不要写.

# 8. 构建基于GRU的解码器 -> 版本1: 无Attention(注意力机制)
class DecoderRNN(nn.Module):
    # 1. 初始化函数.
    def __init__(self, output_size, hidden_size):
        """
        初始化属性信息
        output_size: 解码器输出维度, 即: 目标语言(法语)词汇表大小
        hidden_size: 解码器隐藏层维度, 即: 每个词向量的特征数(256)
        """
        # 1. 初始化父类信息
        super().__init__()
        # 2. 保存输入参数.
        self.output_size = output_size
        self.hidden_size = hidden_size
        # 3. 创建词嵌入层, 输入: [batch_size, seq_len], 输出: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 4. 创建GRU层.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 5. 创建线性层(输出层)
        self.out = nn.Linear(hidden_size, output_size)
        # 6(了解). 创建softmax层.
        self.softmax = nn.LogSoftmax(dim=-1)

    # 2. 前向传播函数.
    def forward(self, input, hidden):
        # 1. 词嵌入处理.
        output = self.embedding(input)
        # 2. ReLU激活函数处理.
        output = F.relu(output)
        # 3. GRU层处理 -> 大白话处理: 都是[1, 1, 256]
        # 输入时: output -> [batch_size, seq_len, hidden_size],   hidden -> [num_layers, batch_size, hidden_size]
        # 输出时: output -> 形状同上,   hidden -> 形状同上
        output, hidden = self.gru(output, hidden)

        # 4. 线性层 和 softmax层处理.
        output = self.softmax(self.out(output[0]))

        return output, hidden

    # 3. 初始化隐藏层输入数据.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# 9. 测试基于GRU的解码器 -> 测试版本1: 无Attention(注意力机制)
def test_decoder(): 
    # 1. 获取数据集加载器对象.
    my_dataloader = get_dataloader()
    # 2. 初始化编码器模型, 并移动到GPU.
    my_encoder_gru = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device)
    print(f'my_encoder_gru: {my_encoder_gru}')
    
    # 3. 初始化解码器模型, 并移动到GPU.
    my_decoder_gru = DecoderRNN(output_size=french_word_n, hidden_size=256).to(device)
    print(f'my_decoder_gru: {my_decoder_gru}')

    # 4. 完整的编码 -> 解码流程测试.
    # 4.1 从数据集加载器中获取1个批次的样本(即: 1条数据)
    for i, (x, y) in enumerate(my_dataloader):
        # 4.2 打印输入数据的信息.
        # print(f'输入数据信息(英语句子): {x.shape}, {x}')  # 输入数据信息(英语句子): torch.Size([1, 6]), tensor([[ 77,  78, 147,  24,   4,   1]])
        # print(f'输入数据信息(法语句子): {y.shape}, {y}')  # 输入数据信息(法语句子): torch.Size([1, 7]), tensor([[123, 297, 126, 246, 384,   5,   1]])
        print(f'输入数据信息(英语句子): {x.shape}')
        print(f'输入数据信息(法语句子): {y.shape}')

        # 4.3 编码过程: 将英文句子编码为 隐藏状态.
        # 4.3.1 初始化编码器.
        h0 = my_encoder_gru.init_hidden()
        # 4.3.2 (编码器)前向传播.
        encoder_output_c, hidden = my_encoder_gru(x, h0)
        

        # 4.4 解码过程: 将隐藏状态解码为法语句子.
        # print(f'观察: 最后1个时间步的output输出: {encoder_output_c[0][-1].shape}') # [6, 256] -> [256]

        # 4.4.1 具体的解码过程 -> 逐个字符生成, 将隐藏状态解码为法语句子.
        # 遍历目标句子(法语句子)的每个时间步.
        for i in range(y.shape[1]):
            # 4.4.2 提取当前时间步的 目标词索引.
            # y[0][i]:     取出batch中第1个样本的第i个词的索引.
            # view(1, -1): 将标量转为[1, 1]的形状, 匹配: 解码器输入要求.
            tmp = y[0][i].view(1, -1)
            # 4.4.3 解码器的前向传播.
            output, hidden = my_decoder_gru(tmp, hidden)
            # 4.4.4 打印解码器的输出信息.
            print(f'第 {i + 1} 个法语单词的预测概率分布: {output.shape}, {output.shape}')

        print('\n' * 5)

# 10. 构建基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制) 
class AttnDecoderRNN(nn.Module):
    # todo 10.1 初始化函数.
    # 参1: 目标(法语)词汇表大小, 参2: 隐藏层维度(和编码器一致), 参3: 随机失活概率, 参4: (句子)最大长度
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 保存输入参数
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 3. 创建词嵌入层.
        # 输入形状: [batch_size, seq_len] -> [1, 1]
        # 输出形状: [batch_size, seq_len, hidden_size] -> [1, 1, 256]
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 4. 注意力权重计算层: 计算查询向量 和 编码器输出的匹配程度.
        # 参1: 拼接后的查询向量 和 隐藏状态 -> [1, 1, 512]
        # 参2: 注意力的权重分布 -> [1, 1, 10]  (最大)10个词
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # 5. 注意力融合层, 将 词嵌入 和 注意力权重进行融合.  
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 6. dropout层, 随机丢弃部分神经元, 防止过拟合.
        self.dropout = nn.Dropout(self.dropout_p)

        # 7. 创建GRU层: 处理序列数据, 维持隐藏状态.
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 8. 输出层: 将GRU的隐藏状态映射为: (法语)目标词汇表大小.
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # 9. 对数softmax()层, 将输出映射为 对数softmax()概率分布
        self.softmax = nn.LogSoftmax(dim=-1)


    # todo 10.2 前向传播函数.
    # 参1: input 当前时间步的输入词索引 -> [batch_size, 1]
    # 参2: hidden 上一个时间步的隐藏状态 -> [1, batch_size, hidden_size]
    # 参3: encoder_outputs 编码器所有时间步的输出 -> [batch_size, seq_len, hidden_size]
    def forward(self, input, hidden, encoder_outputs):
        # 1. 词嵌入层, 输入形状: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        # [1, 1] -> [1, 1, 256]
        embedded = self.embedding(input)

        # 2. 应用Dropout(随机失活层), 随机失活.
        embedded = self.dropout(embedded)

        # 3. 计算注意力权重.
        # step1: torch.cat((embedded[0], hidden[0]), 1))                ->  [1, 512]
        # step2: self.attn(torch.cat((embedded[0], hidden[0]), 1)       ->  [1, 10]
        # step3: 应用softmax()层, 映射为: [1, 10] -> [1, 10]
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 4. 计算注意力上下文.
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # 5.注意力融合层.
        output = torch.cat((embedded[0], attn_applied[0]), 1)       # [1, 1, 512]
        output = self.attn_combine(output).unsqueeze(0)                          # [1, 1, 256]
        # 6. 激活函数处理.
        output = F.relu(output)
        # 7. GRU层: 处理序列数据, 维持隐藏状态.
        output, hidden = self.gru(output, hidden)       # [1, 1, 256]
        # 8. 输出层: 将GRU的隐藏状态映射为: (法语)目标词汇表大小.
        output = self.softmax(self.out(output[0]))      # [1, 4345]

        # 9. 返回结果.
        # 参1: 当前时间步的输出概率分布          ->  [1, 4345]
        # 参2: 更新后的隐藏状态(本次的隐藏状态)   ->  [1, 1, 256]
        # 参3: 注意力权重分布(用于可视化分析, 如果不做, 可以不返回)     -> [1, 10]
        return output, hidden, attn_weights


    # todo 10.3 定义初始化的隐藏状态.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# todo 11. 测试基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制)
def test_attn_decoder():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()

    # 2. 模型初始化阶段
    # 2.1 创建编码器对象.
    # 参1: 英语词汇表大小(2803), 参2: 隐藏层维度
    my_encoder = EncoderRNN(english_word_n, 256).to(device)

    # 2.2 创建解码器对象.
    # 参1: 法语词汇表大小(4345), 参2: 隐藏层维度
    my_decoder = AttnDecoderRNN(french_word_n, 256).to(device)

    # 3. 模型训练(推理)阶段
    # 3.1 从数据加载器中获取1个样本.
    for i, (x, y) in enumerate(my_dataloader):
        # 3.2 打印输入信息
        print(f'x(英语句子): {x.shape}, {x}')       # torch.Size([1, 6]),
        print(f'y(法语句子): {y.shape}, {y}')       # torch.Size([1, 8]),

        # 3.3 编码过程 -> 将英语句子 编码成 隐藏状态 -> (中间语义张量C)
        hidden = my_encoder.init_hidden()
        # output: 所有时间步的隐藏状态: [1, seq_len, 256]
        # hidden: 最后一个时间步的隐藏状态: [1, 1, 256]
        output, hidden = my_encoder(x, hidden)

        # 3.4 准备编码器的输出 -> 用于Attention机制.
        # 创建1个固定大小的张量, 用于存储编码器输出, 形状为: [10, 256]
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)

        # 3.5 将编码器实际输出 复制到 固定大小的张量中.
        for idx in range(output.shape[1]):
             encoder_output_c[idx] = output[0, idx]

        # 3.6 解码过程: 将隐藏状态 解码为 法语句子(必须逐词翻译)
        # 3.6.1 遍历目标句子的每个时间步.
        for i in range(y.shape[1]):
            # 3.6.2 提取当前时间步的目标词索引.
            tmp = y[0][i].view(1, -1)       # 例如: [[1595]]

            # 3.6.3 执行解码器的前向传播.
            # (实际)参数列表解释
            # tmp: 当前时间步的输入词索引, 形状: [1, 1]
            # hidden: 上一个时间步的隐藏状态, 形状: [1, 1, 256]
            # encoder_output_c: 编码器(所有时间步)的输出, 形状: [10, 256]

            # 返回值参数列表解释:
            # output: 当前时间步的输出概率分布, 形状: [1, 4345]
            # hidden: 当前时间步的隐藏状态, 形状: [1, 1, 256]
            # attn_weights: 当前时间步的注意力权重分布, 形状: [1, 10]

            #                                          Q    K         V
            output, hidden, attn_weights = my_decoder(tmp, hidden, encoder_output_c)

            # 3.6.4 打印结果.
            print(f'解码output.shape: {output.shape}')    # [1, 4345]
            print(f'解码hidden.shape: {hidden.shape}')    # [1, 1, 256]
            print(f'解码attn_weights.shape: {attn_weights.shape}')    # [1, 10]
            print('\n' * 3)

        # 只看1个句子(样本)
        break

# 12. 构建模型内部迭代训练函数 -> 即: 完成单批次的训练过程
# 12.1 定义模型训练参数
# 学习率, 训练轮数, Teacher_Forcing比例, 输出信息打印间隔(每训练1000条打印一次),  绘图间隔(每训练100条绘图一次)
my_lr, epochs, teacher_forcing_ratio, print_interval_num, plot_interval_num = 1e-4, 1, 0.5, 1000, 100

# 12.2 定义函数, 实现: 单批次训练, 完成1个样本的 编码 -> 解码 -> 反向传播 -> 优化参数...
def train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss):
    """
    函数作用, 实现: 单批次训练, 完成1个样本的 编码 -> 解码 -> 反向传播 -> 优化参数...
    x: 输入序列, 即: 英语句子, 形状为: [batch_size = 1, seq_len]
    y: 目标序列, 即: 法语句子, 形状为: [batch_size = 1, seq_len]
    my_encoder_rnn: 编码器对象
    my_attn_decoder_rnn: 解码器对象(带注意力机制)
    myadam_encode: 编码器优化器
    myadam_decode: 解码器优化器
    my_crossentropy_loss: 损失函数
    """
    # 1. 编码阶段, 将输入的序列转换为上下文向量, 初始的隐藏状态: [1, 1, 256]
    encoder_hidden = my_encoder_rnn.init_hidden()

    # 编码器的前向传播.
    encoder_output, encoder_hidden = my_encoder_rnn(x, encoder_hidden)

    # 2. 解码参数准备
    # 2.1 构建编码器的输出张量, 用于: 注意力计算, 形状为: [10, 256]
    encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
    # 2.2 复刻实际的编码器输出 -> 固定长度张量, 即: [假设6个单词, 256] -> [10, 256]
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]

    # 2.3 解码器初始化隐藏状态.
    decoder_hidden = encoder_hidden     # [1, 1, 256]

    # 2.4 解码器的初始输入.
    input_y = torch.tensor([[SOS_token]], device=device)        # [1, 1]

    # 3. 初始化损失值
    my_loss = 0.0
    y_len = y.shape[1]      # 目标序列长度(要预测的法语句子长度), 例如: 9

    # 4. 根据概率值决定是否用 Teacher Forcing.
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # 4.1 走这里, 说明使用 Teacher Forcing, 就: 用真实标签作为下一步的输入.
        for i in range(y_len):
            # 4.1.1 解码器的前向传播
            # 输入: input_y -> [1, 1], decoder_hidden -> [1, 1, 256], encoder_output_c -> [10, 256]
            # 输出: output_y -> [1, 4345], decoder_hidden -> [1, 1, 256], attn_weights -> [1, 10]
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # 4.1.2 获取当前时间步的真实标签.
            target_y = y[0][i].view(1)
            # 4.1.3 累加损失
            my_loss += my_crossentropy_loss(output_y, target_y)
            # 4.1.4 下个时间步的输入 直接使用 真实标签.
            input_y = y[0][i].view(1, -1)   # [1, 1]
    else:
        # 4.2 非Teacher Forcing, 就: 用上一时间步的预测结果作为下一步的输入.
        for i in range(y_len):
            # 4.2.1 解码器前向传播
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # 4.2.2 获取当前时间步的真实标签.
            target_y = y[0][i].view(1)
            # 4.2.3 累加损失
            my_loss += my_crossentropy_loss(output_y, target_y)
            # 4.2.4 获取预测的下一个词, 即: 获取概率最高的词的索引和概率.
            topv, topi = output_y.topk(1)
            # 4.2.5 如果预测到句子的结束标记, 则停止预测.
            if topi.squeeze().item() == EOS_token:
                break
            # 4.2.6 走到这里, 说明没有预测到结束标记, 则: 将预测的词 作为 下一步的输入.
            input_y = topi.detach()     # [1, 1]

    # 5. 反向传播和参数更新.
    myadam_encode.zero_grad()       # 梯度清零, 编码器.
    myadam_decode.zero_grad()       # 梯度清零, 解码器.

    my_loss.backward()              # 反向传播.
    myadam_encode.step()            # 参数更新, 编码器.
    myadam_decode.step()            # 参数更新, 解码器.

    # 6. 返回平均损失.
    return my_loss.item() / y_len

# 13. 构建模型训练函数 -> 即: 完成所有批次的训练过程, 即: 多轮, 多批次训练过程.
def train_seq2seq():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()
    # 2. 模型初始化, 记得把模型移动到GPU上, 我的电脑(训练1轮): CPU训练时间 50分钟, GPU训练时间: 20分钟
    # 2.1 编码器, 输入维度 = 英文词汇表大小2803, 隐藏层维度: 256
    my_encoder_rnn = EncoderRNN(english_word_n, 256).to(device)
    # 2.2 解码器, 输入维度 = 法语词汇表大小4345, 隐藏层维度: 256
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256, 0.1, 10).to(device)

    # 3. 优化器初始化, 使用: Adam优化器, 学习率: 1e-4
    myadam_encode = optim.Adam(my_encoder_rnn.parameters(), lr=my_lr)       # 编码器优化器
    myadam_decode = optim.Adam(my_attn_decoder_rnn.parameters(), lr=my_lr)  # 解码器优化器

    # 4. 损失函数初始化, 使用: NLLLoss
    my_crossentropy_loss = nn.NLLLoss()

    # 5. 训练参数初始化.
    plot_loss_list = []         # 存储绘图用的损失值.

    # 6. 具体的多轮, 多批次训练过程.
    # 6.1 外层循环, 控制训练轮数.
    for epoch_idx in range(1, epochs + 1):
        # 6.1.1 初始化本轮的损失累加器
        print_loss_total, plot_loss_total = 0.0, 0.0
        # 6.1.2 记录本轮开始训练时间.
        start_time = time.time()

        # 6.2 内层循环, 遍历数据集的每个样本(即: 每轮具体的 所有批次训练过程)
        for item, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            # 6.2.1 调用内部训练函数, 完成: 单批次(单样本)的训练过程.
            myloss = train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss)

            # 6.2.2 累加损失.
            print_loss_total += myloss
            plot_loss_total += myloss

            # 6.3 打印训练日志(每 print_interval_num=1000 个样本打印一次)
            if item % print_interval_num == 0:
                # 计算平均损失.
                print_loss_avg = print_loss_total / print_interval_num
                # 重置损失累加器.
                print_loss_total = 0.0
                # 打印训练信息: 轮次, 平均损失, 耗时.
                print(f'轮次: {epoch_idx}, 平均损失: {print_loss_avg:.4f}, 耗时: {time.time() - start_time:.4f} s(秒)!')

            # 6.4 记录损失用于绘图(每 plot_interval_num=100 个样本记录一次)
            if item % plot_interval_num == 0:
                # 计算平均损失.
                plot_loss_avg = plot_loss_total / plot_interval_num
                # 存储损失值.
                plot_loss_list.append(plot_loss_avg)
                # 重置损失累加器.
                plot_loss_total = 0.0

            # 扩展: 每轮训练3000个样本后, 结束训练, 实际开发, 这个代码万万不能写.
            # if item > 3000:
            #    break

        # 6.3 走到这里, 说明一轮训练完毕, 保存模型.
        torch.save(my_encoder_rnn.state_dict(), f'./model/my_encoder_rnn_{epoch_idx}.pth')  # pickle文件后缀, .pth, .pkl, .pickle
        torch.save(my_attn_decoder_rnn.state_dict(), f'./model/my_attn_decoder_rnn_{epoch_idx}.pth')

    # 7. 训练结束后, 绘制损失曲线.
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/Seq2Seq_loss.png')
    plt.show()

    # 8. 训练结束, 返回结果.
    return plot_loss_list           # 训练损失列表(每训练100个样本的平均损失)

# todo 14. 构建模型评估 -> 用训练好的seq2seq模型进行 翻译.
def evaluate_seq2seq(x, my_encoder_rnn, my_attn_decoder_rnn):   # 英语句子, 编码器, 解码器(带注意力机制)
    # 0. 关闭梯度计算, 节省内存并加速推理 -> 只适用于 模型的预测过程.
    with torch.no_grad():
        # 1.编码阶段: 将输入的英文句子 -> 隐藏状态.
        encode_hidden = my_encoder_rnn.init_hidden()
        # 本次的输出, 本次的隐藏状态 = 编码器模型(本次的输入, 上一时刻的隐藏状态)  
        encode_output, encode_hidden = my_encoder_rnn(x, encode_hidden)

        # 2. 解码器参数准备.
        # 2.1 构建固定长度的编码器输出张量.
        encode_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
        for idx in range(x.shape[1]):
            encode_output_c[idx] = encode_output[0, idx]

        # 2.2 解码器的隐藏状态. 
        decode_hidden = encode_hidden

        # 2.3 解码器的初始输入, 句子的开始标记.
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3. 自回归解码过程(逐个生成目标句子)
        # 定义遍历, 记录: 存储解码后的法语单词
        decode_words = []
        # 初始化注意力矩阵
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        # 遍历, 开始解码.
        for idx in range(MAX_LENGTH):
            # 3.1 解码器, 前向传播.
            # 输入: 当前输入词的索引(Q), 解码隐藏状态(K), 编码器输出张量(V))
            # 输出: 下个词的概率分布, 更新后的隐藏状态, 注意力权重分布矩阵
            output_y, decode_hidden, attn_weights = my_attn_decoder_rnn(input_y, decode_hidden, encode_output_c)

            # 3.2 记录注意力矩阵.
            decoder_attentions[idx] = attn_weights
            # 3.3 预测下个词.
            topv, topi = output_y.topk(1)
            # 3.4 处理终止条件, 如果预测到EOS标记, 结束生成.
            if topi.squeeze().item() == EOS_token:
                break
            else:
                # 3.5 否则, 则添加预测的词到结果列表.
                decode_words.append(french_index2word[topi.squeeze().item()])
            # 3.6 更新输入: 把当前预测词作为下个时间步的输入.
            input_y = topi.detach()

        # 4. 返回解码结果 和 注意力矩阵.
        return decode_words, decoder_attentions[:idx + 1]


# todo 15. 模型评估函数调用, 记载模型, 对比 自定义样本进行 翻译.
# 模型路径
PATH1 = './model/my_encoder_rnn_1.pth'
PATH2 = './model/my_attn_decoder_rnn_1.pth'

# 定义函数
def test_seq2seq_evaluate():
    # 1. 获取数据加载器对象
    my_dataloader = get_dataloader()
    # 2. 加载编码器模型.
    my_encoder_rnn = EncoderRNN(english_word_n, hidden_size=256).to(device)

    # map_location: 确保在CPU和GPU(Cuda)都能加载. 正常应该是用GPU训练, 就用GPU预测, 用CPU训练, 就用CPU预测
    # 写了 map_location能实现: 用GPU训练, 用CPU预测
    # weights_only: 只加载模型权重参数
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), False)
    print(f'my_encoder_rnn编码器模型架构: {my_encoder_rnn}')

    # 3. 加载解码器模型
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), False)
    print(f'my_attn_decoder_rnn解码器模型架构: {my_attn_decoder_rnn}')

    # 4. 自定义测试样本.
    my_sample_pairs = [
        # 格式: ['英文句子', '法语句子']
        ['i m the spokesperson for this organization .', 'je suis le porte parole de cette institution .'],
        ['i am thinking about buying a new parasol .', 'je songe a acheter un nouveau parasol .'],
        ['he is able to swim very fast .', 'il est capable de nager tres vite .']
    ]
    print(f'自定义测试样本: {my_sample_pairs}')

    # 5. 对每个样本进行翻译.
    for index, pair in enumerate(my_sample_pairs):
        x = pair[0]     # 英语句子
        y = pair[1]     # 法语句子

        # 5.1 文本数值化, 英语句子 -> 索引序列
        tmpx = [english_word2index[word] for word in x.split(' ')]
        tmpx.append(EOS_token)      # 添加结束标记.
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 5.2 模型预测.
        decode_words, attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)

        # 5.3 把翻译结果转成句子格式.
        output_sentence = ' '.join(decode_words)
        print(f'输入(原始英文句子): {x}')
        print(f'输入(原始法语句子): {y}')
        print(f'输出(翻译后的法语句子): {output_sentence}')
        print(' -.- ' * 10, '\n')


# todo 16. 绘制注意力图的函数 -> Attention张量绘图(看看即可, 无需编写)
def test_attention():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()
    # 2. 实例化模型.
    # 2.1 编码器模型
    my_encoder_rnn = EncoderRNN(english_word_n, 256).to(device)
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), False)

    # 2.2 解码器模型.
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), False)

    # 3. 定义遍历, 记录: 英语句子
    sentence = 'we re both teachers .'
    # 对上述的样本进行数值化.
    tmpx = [english_word2index[word] for word in sentence.split(' ')]
    tmpx.append(EOS_token)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

    # 4. 模型预测.
    decode_words, attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)
    print(f'decode_words: {decode_words}')


    # 5. 绘制注意力图.
    plt.matshow(attentions.numpy())     # 以矩阵列表的形式 显示.  matrix: 矩阵.
    # 保存图像.
    plt.savefig('./img/s2s_attn.png')
    plt.show()

    # 6. 打印下参数.
    print(f'attentions.numpy(): {attentions.numpy()}')
    print(f'attentions.size(): {attentions.size()}')

# todo n.测试代码
if __name__ == '__main__':
    
    # 1. 测试用例: 包括多种语言和特殊符号的字符串.
    # s = ' I Love You! .?! 我#爱你 end'
    # print(f'(原始的字符串)s: {s}')
    # print(' -.- ' * 10)
    # normalizeString(s)

    # 2. 测试数据预处理函数.
    # english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()
    # print(f'英语词汇表映射: {english_word2index}')
    # print(f'英语单词反向映射: {english_index2word}')
    # print(f'英语词汇表大小: {english_word_n}')
    # print(f'法语词汇表映射: {french_word2index}')
    # print(f'法语单词反向映射: {french_index2word}')
    # print(f'法语词汇表大小: {french_word_n}')
    # print(f'双语句子对: {my_pairs[:5]}')

    # 3. 测试数据加载器.
    # get_dataloader()

    # 4. 测试基于GRU的编码器.
    # test_encoder()

    # 5. 测试基于GRU的解码器 -> 测试版本1: 无Attention(注意力机制)
    # test_decoder()

    # 6. 测试基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制)
    # test_attn_decoder()

    # 7. 模型训练.
    # train_seq2seq()

    # 8. 模型评估.
    test_seq2seq_evaluate()

    # 9. 绘制注意力图.
    # test_attention()

```



效果图如下：

![image-20260910201159629](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260910201159629.png)



 

# 四、Transformer架构

## 4.1 transformer架构介绍

### 4.1.1 模型作用

Transformer 是一种完全基于注意力机制（Attention Mechanism）的序列到序列（Seq2Seq）模型。它摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN），仅使用自注意力（Self-Attention）和前馈神经网络来建模序列数据。

Transformer 的主要作用包括：

- **机器翻译**：将一种语言翻译为另一种语言。
- **文本生成**：如 GPT 系列，基于自回归生成文本。
- **文本理解**：如 BERT，用于分类、问答、序列标注等。
- **多模态任务**：如 ViT（Vision Transformer）处理图像，CLIP 处理图文匹配。

其核心优势在于：

1. **并行计算**：不同于 RNN 需要按时间步顺序计算，Transformer 可以并行处理整个序列，大幅提升训练效率。
2. **长距离依赖建模**：自注意力机制可以直接建模序列中任意两个位置之间的关系，不受距离限制。
3. **可扩展性**：结构统一，易于堆叠，适合大规模预训练。

### 4.1.2 总架构介绍

Transformer 的整体架构由**编码器（Encoder）**、**解码器（Decoder）、输入部分、输出部分** 四部分组成，每部分都由多个相同的层堆叠而成。

架构图如下：

![image-20260911140550119](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911140550119.png)





**编码器层（Encoder Layer）**包含两个子层：

1、多头自注意力（Multi-Head Self-Attention）

![image-20260911145118638](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911145118638.png)

2、前馈神经网络（Feed Forward Network）

![image-20260911145130609](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911145130609.png)

每个子层都配有残差连接（Residual Connection）和层归一化（Layer Normalization）。

由N(6)个编码器层堆叠而成

**解码器层（Decoder Layer）**包含三个子层：

1、掩码多头自注意力（Masked Multi-Head Self-Attention）

![image-20260911145555051](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911145555051.png)

2、多头交叉注意力（Multi-Head Cross-Attention），Query 来自解码器，Key 和 Value 来自编码器

![image-20260911145604383](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911145604383.png)

3、前馈神经网络

![image-20260911145619439](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911145619439.png)

同样，每个子层都配有残差连接和层归一化。

**Transformer 论文中的超参数**：

- 编码器和解码器层数：N = 6
- 模型维度（d_model）：512
- 前馈网络内部维度（d_ff）：2048
- 注意力头数（h）：8
- Dropout：0.1

### 4.1.3 注意点

1. **位置编码**：由于 Transformer 不包含循环结构，无法感知序列顺序，因此必须显式添加位置编码（Positional Encoding）。
2. **掩码机制**：解码器的自注意力必须使用因果掩码（Causal Mask），防止当前位置看到未来的信息。编码器和解码器的填充位置需要用 Padding Mask 忽略。
3. **多头注意力**：将 Query、Key、Value 分成多个头，分别计算注意力，再拼接，能够捕捉不同子空间的信息。
4. **残差连接与层归一化**：有助于缓解梯度消失，加速训练。
5. **权重共享**：输入嵌入、输出嵌入和输出线性层可以共享权重（论文中采用此策略）。
6. **缩放点积注意力**：除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 梯度消失。



### 4.1.4 流程

![image-20260911152507211](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911152507211.png)

注意力机制抓关联，多层堆叠挖深度，残差归一化稳训练

**流程：**

输入文字数据`input` ==> 词嵌入层 `Input Embedding` ==> 位置编码`Positional Encoding` ==>  编码器多层加工(多头自注意力`Multi-Head Self-Attention` + 前馈`Feed Forward`)  ==> 解码器多层加工(掩码注意力 + 关联编码器 + 前馈) ==> 线性层 ==> `softmax` ==> 输出概率并选词



**输入部分：**

`Input Embedding`： 将输入的文字 转换成 数值向量（词向量）

`Positional Encoding`：Transformer本身不明白词的顺序，所以需要给词向量添加“位置信息”，告诉模型谁前谁后

**编码器：**

`Multi-Head Self-Attention`：多头注意力层，让模型同时从多个角度 关注 句子中的词

`Add & Norm`：残差连接 + 规范化层。Add 是把注意力层的输出和输入加在一起，防止信息丢失太多，Norm是把数据归一化（把数据限制到一个范围），避免训练时发生跑偏（梯度消失，梯度爆炸）

`Feed Forward`：前馈全连接层，对每个位置的词向量**单独强化特征**

`编码器Nx`：重复堆叠，将上述**多头自注意力、Add & Norm、前馈网络**重复N次（论文中说明是6次）

**输入部分：**

`Output Embedding`：将输出的目标 转换成 数值向量（词向量）

`Positional Encoding`：给输出词向量添加“位置信息”，让模型明确先后顺序

**解码器：**

`Masked Multi-Head Self-Attention`：掩码多头注意力，防止偷看“未来的词”

`Multi-Head Self-Attention`：多头注意力层，让解码器关注编码器输出的内容。

`Add & Norm`、`Feed Forward`、`NX`：均和编码器中一样

 输出部分：

`Linear`：线性层，把解码器输出的向量，调整成 vocab_size词汇表大小 维度

`Softmax`：激活层，将线性层的输出的数值，转成“概率”总和为1

 

## 4.2 输入部分实现

### 4.2.1 输入部分介绍

Transformer 的输入部分负责将原始文本序列转换为模型可以处理的数值张量，并注入位置信息。输入部分包括：

1. 源文本嵌入层 及其 位置编码器
1. 目标文本嵌入层 及其 位置编码器

![image-20260911194613050](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260911194613050.png)



### 4.2.2 文本嵌入层

作用：文本嵌入层将离散的词索引转换为连续的向量表示。Transformer 中的嵌入层通常与词向量维度一致

**API**：`torch.nn.Embedding(num_embeddings, embedding_dim)`

**常用参数表**：

|      参数名      |    含义    |                      作用                      |
| :--------------: | :--------: | :--------------------------------------------: |
| `num_embeddings` | 词汇表大小 |        指定嵌入矩阵的行数，即有多少个词        |
| `embedding_dim`  |  嵌入维度  |                每个词向量的维度                |
|  `padding_idx`   |  填充索引  | 指定哪个索引为填充符，其对应向量不参与梯度更新 |
|    `_weight`     | 自定义权重 |             可用于加载预训练词向量             |

**代码实现**：

```python
import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: [batch_size, seq_len]
        # 输出: [batch_size, seq_len, d_model]
        return self.lut(x) * math.sqrt(self.d_model)
```

**说明**：乘以 $\sqrt{d_{model}}$ 是为了缩放嵌入向量，使其与位置编码的量级匹配。



### 4.2.3 位置编码层

由于 Transformer 没有循环结构，必须通过**位置编码**注入序列中**每个位置**的信息。位置编码需要满足以下需求：

1. **唯一性**：每个位置应有唯一的编码。
2. **有界性**：编码值应在合理范围内，避免数值不稳定。
3. **可外推性**：能够处理比训练时更长的序列。
4. **相对位置可表示**：模型应能通过编码容易地学到相对位置关系。
5. **无需额外训练参数**：最好不引入需要训练的参数。

正弦和余弦函数恰好能满足以上所有需求。

位置编码的定义为：

对于当前元素位置 $pos$ 和当前向量的第 $i$维度：

偶数维度：
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

奇数维度：
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中 $pos$ 是词在句子中的索引位置，$i$ 是维度的索引，$PE$就是位置编码(Position Encoding)，$d_{model}$表示嵌入向量的维度

计算完位置编码后还要和 词向量进行向量相加。

#### 4.2.3.1 了解——为什么选择正弦和余弦函数

**1. 有界性**

正弦和余弦函数的值域为 $[-1, 1]$，因此位置编码的每个维度都是有限的。这与词嵌入相加后不会破坏词嵌入的数值范围，有利于训练稳定。

**2. 不同维度不同频率**

对于不同的维度 $i$，频率为：

$$
\omega_i = \frac{1}{10000^{2i/d_{model}}}
$$

- 当 $i$ 越小时，$\omega_i$ 越大，由 $T = 2Π/w$ ，周期越小。对应高频变化，能区分相邻位置。
- 当 $i$ 越大时，$\omega_i$ 越小，由 $T = 2Π/w$ ，周期越大。对应低频变化，能区分远距离位置。

这使得位置编码在不同维度上捕捉不同尺度的位置信息，类似于二进制编码中不同位代表不同数量级。

**3. 相对位置可线性表示**

这是使用正弦和余弦函数的**最重要原因**。对于任意固定的偏移量 $k$，位置 $pos+k$ 的编码可以表示为位置 $pos$ 编码的线性函数。

证明：

利用三角恒等式：

$$
\sin(\alpha + \beta) = \sin\alpha \cos\beta + \cos\alpha \sin\beta
$$

$$
\cos(\alpha + \beta) = \cos\alpha \cos\beta - \sin\alpha \sin\beta
$$

设 $\omega_i = \frac{1}{10000^{2i/d_{model}}}$，则：

$$
PE_{(pos+k, 2i)} = \sin(\omega_i (pos+k)) = \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)
$$

$$
PE_{(pos+k, 2i+1)} = \cos(\omega_i (pos+k)) = \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)
$$

将这两个式子写成矩阵形式：

$$
\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix}
=
\begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix}
\begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}
$$

这是一个**旋转矩阵**，其参数仅依赖于偏移量 $k$，与绝对位置 $pos$ 无关。这意味着：

- 模型可以通过位置编码的线性变换轻松学习到相对位置关系。
- 对于任意固定的相对距离 $k$，编码之间的关系是确定的、可学习的。

这对注意力机制尤为重要，因为注意力本质上是在比较 Query 和 Key 之间的关系。如果位置编码能够表达相对位置，模型就更容易学到“关注前一个词”或“关注后两个词”这样的模式。

**4. 可外推性**

由于正弦和余弦是确定性函数，即使序列长度超过训练时的最大长度，也可以继续计算位置编码，而不会像可学习的位置嵌入那样遇到未知位置的问题。虽然实际外推效果可能有限，但至少提供了数学上的可能性。

**5. 无需额外参数**

正弦和余弦位置编码是固定的、非参数化的，不需要在训练中学习。这减少了模型参数量，也避免了过拟合风险。



#### 4.2.3.2 了解——为什么位置编码在 sin/cos 公式下能保证每个位置是唯一的

位置编码的每一维都是一个周期函数，但**每个维度的周期不同**。

- 第 0 维（$i=0$）：频率为 $\omega_0 = 1$，周期为 $2\pi$，变化最快。
- 第 1 维（$i=1$）：频率为 $\omega_1 = 1/10000^{2/d_{model}}$，周期更长。
- 第 $i$ 维：频率为 $\omega_i = 1/10000^{2i/d_{model}}$，周期为 $2\pi / \omega_i$，随 $i$ 增大而急剧变长。

由于每个维度的周期不同，组合起来就像一个**多频率时钟系统**。类比：秒针、分针、时针各自周期不同，组合起来能唯一标识一天中的每个时刻。类似地，不同维度的正弦余弦组合起来，可以唯一标识序列中每个位置。



**1. 将位置编码视为从 $pos$ 到高维向量的映射**

将位置编码写成一个向量：

$$
PE_{pos} = \begin{pmatrix} \sin(\omega_0 pos) \\ \cos(\omega_0 pos) \\ \sin(\omega_1 pos) \\ \cos(\omega_1 pos) \\ \vdots \\ \sin(\omega_{d/2-1} pos) \\ \cos(\omega_{d/2-1} pos) \end{pmatrix}
$$

其中 $\omega_i = \frac{1}{10000^{2i/d_{model}}}$，且 $\omega_0 > \omega_1 > \cdots > \omega_{d/2-1}$。



**2. 关键：频率的层级结构**

这些频率按照几何级数递减：

$$
\omega_i = 10000^{-2i/d_{model}}
$$

例如当 $d_{model} = 512$ 时：

- $i=0$：$\omega_0 = 1$
- $i=1$：$\omega_1 \approx 0.964$
- $i=2$：$\omega_2 \approx 0.930$
- ...
- $i=255$：$\omega_{255} \approx 10^{-4}$

最高频率与最低频率之间相差约 $10000$ 倍。这种频率的跨度保证了即使 $pos$ 非常大，至少有一部分维度仍在快速变化，能够区分相邻位置。



**3. 唯一性的论证**

假设存在两个不同位置 $pos_1 \neq pos_2$，使得它们的编码完全相同：

$$
PE_{pos_1} = PE_{pos_2}
$$

这意味着对于所有 $i$：

$$
\sin(\omega_i pos_1) = \sin(\omega_i pos_2) \quad \text{且} \quad \cos(\omega_i pos_1) = \cos(\omega_i pos_2)
$$

由三角函数的周期性，$\sin$ 和 $\cos$ 同时相等当且仅当：

$$
\omega_i (pos_1 - pos_2) = 2\pi k_i, \quad k_i \in \mathbb{Z}
$$

即：

$$
pos_1 - pos_2 = \frac{2\pi k_i}{\omega_i}
$$

对每个 $i$ 都成立。也就是说，差值 $pos_1 - pos_2$ 必须是所有 $\frac{2\pi}{\omega_i}$ 的整数倍。

但是，不同的 $\omega_i$ 之间的比值是无理数（例如 $\omega_0 / \omega_1 = 10000^{2/d_{model}}$ 通常不是有理数），因此它们的最小公倍数在实数意义上不存在（除非差值为 0）。这意味着只有当 $pos_1 = pos_2$ 时，所有维度才能同时相等。

因此，位置编码在合理范围内是唯一的。



#### 4.2.3.3 了解——位置编码与词向量相加后，如何保证唯一性？

位置编码 $PE_{pos}$ 是唯一的，词嵌入 $E_{word}$ 也是唯一的（每个词对应一个向量）

但模型实际输入是两者相加：
$$
h = E_{word} + PE_{pos}
$$

问题：两个不同的 $(word, pos)$ 对，会不会相加后得到相同的 $h$？  

即：是否存在 $E_{a} + PE_{1} = E_{b} + PE_{2}$ 但 $(a,1) \neq (b,2)$

从数学上讲，**无法保证**相加后一定唯一。  

因为如果 $E_a - E_b = PE_2 - PE_1$，那么两者相加结果相同。

但这种情况在实际中**极难发生**，原因如下：

**1. 高维空间的稀疏性**

词嵌入和位置编码都是 $d_{model}$ 维向量（如 512 维）。  

在高维空间中，两个随机向量几乎总是近似正交的，向量之间的差异非常大。  

要使 $E_a - E_b = PE_2 - PE_1$ 精确成立，需要非常巧合的数值关系，概率极低。

**2. 词嵌入是可学习的**

词嵌入不是固定的，而是在训练中不断调整的。  

模型会自动将词嵌入放置到合适的位置，使得相加后的表示具有区分度。  

如果某种相加方式导致冲突，训练过程中的梯度会推动词嵌入调整，避免碰撞。

**3. 位置编码是固定的、结构化的**

位置编码具有确定的结构（不同频率的正弦余弦），其分布与随机初始化的词嵌入差异很大。  

两者相加后，位置信息以特定模式“叠加”在词嵌入上，模型可以通过注意力机制和后续层轻松分离出位置和语义信息。



##### 为什么选择相加而不是拼接？

拼接（Concatenation）可以严格保证唯一性，因为 $[E_{word}; PE_{pos}]$ 不会与 $[E_{other}; PE_{other}]$ 混淆。  

但 Transformer 选择了相加，原因：

| 方法 | 优点                       | 缺点                                 |
| ---- | -------------------------- | ------------------------------------ |
| 相加 | 维度不变，计算高效，参数少 | 理论上不保证唯一                     |
| 拼接 | 严格唯一，信息不混合       | 维度翻倍，后续层参数增加，计算量增大 |

相加后维度保持 $d_{model}$，使得模型可以堆叠更多层而不增加维度。  而

拼接会使维度变为 $2 d_{model}$，需要额外的线性层降维，增加参数量和计算量。



#### 4.2.3.4 示例代码

代码实现：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

**说明**：`register_buffer` 将 `pe` 注册为缓冲区，不参与梯度更新，但会随模型保存和加载。

---

## 4.3 编码器部分实现

### 4.3.1 掩码张量

#### 概念

掩码（Mask）用于在注意力计算中屏蔽某些位置，防止模型关注到无效信息。Transformer 中常用两种掩码：

1. **Padding Mask**：屏蔽填充位置（`<pad>`），使注意力权重不分配给这些位置。
2. **Causal Mask（Sequence Mask）**：在解码器自注意力中，防止当前位置看到未来的词。

**代码实现**：

```python
def subsequent_mask(size):
    """生成因果掩码，形状 [1, size, size]，上三角为 0，下三角和对角线为 1"""
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0  # 返回布尔张量，True 表示可见

def padding_mask(seq, pad_idx=0):
    """生成填充掩码，形状 [batch_size, 1, seq_len]"""
    return (seq != pad_idx).unsqueeze(1)
```

**说明**：

- 因果掩码确保解码器在生成第 $t$ 个词时只能看到前 $t$ 个位置。
- 填充掩码忽略填充位置，避免它们影响注意力计算。

### 4.3.2 注意力机制

#### 概念

注意力机制在 Transformer 中采用缩放点积注意力（Scaled Dot-Product Attention）。

**公式**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

其中：

- $Q$：查询矩阵，形状 `[batch, n, d_k]`
- $K$：键矩阵，形状 `[batch, m, d_k]`
- $V$：值矩阵，形状 `[batch, m, d_v]`
- $d_k$：键的维度

**代码实现**：

```python
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

### 4.3.3 多头注意力

#### 概念

多头注意力将 Query、Key、Value 分别通过多个线性变换投影到不同的子空间，然后并行计算注意力，最后拼接并线性变换。

**公式**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

**PyTorch API**：`torch.nn.MultiheadAttention`

**常用参数表**：

| 参数名        | 含义         | 作用                                          |
| ------------- | ------------ | --------------------------------------------- |
| `embed_dim`   | 模型维度     | 输入和输出的特征维度                          |
| `num_heads`   | 注意力头数   | 将模型维度分成多少个头                        |
| `dropout`     | Dropout 概率 | 注意力权重的 dropout                          |
| `batch_first` | 批次优先     | 若为 True，输入形状为 `(batch, seq, feature)` |
| `bias`        | 是否使用偏置 | 线性层是否包含偏置                            |

**代码实现**：

```python
import copy
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1. 线性变换并分头: [batch, seq, d_model] -> [batch, h, seq, d_k]
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2. 计算注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. 拼接并线性变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

### 4.3.4 前馈连接层

#### 概念

前馈神经网络（Feed Forward Network, FFN）对每个位置独立应用相同的全连接层。它由两个线性变换和一个激活函数组成。

**公式**：

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

**代码实现**：

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

### 4.3.5 规范化层

#### 概念

层归一化（Layer Normalization）对每个样本的特征维度进行归一化，稳定训练。

**公式**：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

**PyTorch API**：`torch.nn.LayerNorm`

**常用参数表**：

| 参数名               | 含义         | 作用                                        |
| -------------------- | ------------ | ------------------------------------------- |
| `normalized_shape`   | 归一化维度   | 指定对哪些维度进行归一化，通常为 d_model    |
| `eps`                | 数值稳定性   | 防止除以零的小常数，默认 1e-5               |
| `elementwise_affine` | 是否仿射变换 | 若为 True，使用可学习的 $\gamma$ 和 $\beta$ |

**代码实现**：

```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### 4.3.6 子层连接结构

#### 概念

子层连接（Sublayer Connection）将每个子层（如注意力、前馈网络）包裹起来，加入残差连接和层归一化。

**公式**：

$$
\text{Sublayer}(x) = \text{LayerNorm}(x + \text{Dropout}(\text{Sublayer}(x)))
$$

**代码实现**：

```python
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

### 4.3.7 编码器层

#### 概念

编码器层由两个子层连接组成：

1. 多头自注意力
2. 前馈神经网络

**代码实现**：

```python
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

**编码器整体**：

```python
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

---

## 4.4 解码器部分实现

### 4.4.1 解码器介绍

解码器同样由 N 个相同的层堆叠而成，每层包含三个子层：

1. **掩码多头自注意力**：防止看到未来信息。
2. **多头交叉注意力**：Query 来自解码器，Key 和 Value 来自编码器输出。
3. **前馈神经网络**。

每个子层都配有残差连接和层归一化。

### 4.4.2 解码器层

**代码实现**：

```python
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([copy.deepcopy(SublayerConnection(size, dropout)) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        # 1. 掩码自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2. 交叉注意力
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        # 3. 前馈网络
        return self.sublayer[2](x, self.feed_forward)
```

### 4.4.3 解码器

**代码实现**：

```python
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

---

## 4.5 模型构建

### 4.5.1 模型构建介绍

将编码器、解码器、输入嵌入、位置编码和输出层组合起来，形成完整的 Transformer 模型。

### 4.5.2 编码器-解码器结构实现

**代码实现**：

```python
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```

### 4.5.3 Transformer模型构建

**代码实现**：

```python
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(src_vocab, d_model), c(position)),
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(position)),
        Generator(d_model, tgt_vocab)
    )
    # 参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
```

**常用参数表**：

| 参数名      | 含义               | 作用                     |
| ----------- | ------------------ | ------------------------ |
| `src_vocab` | 源语言词汇表大小   | 输入嵌入层的维度         |
| `tgt_vocab` | 目标语言词汇表大小 | 输出嵌入层和生成器的维度 |
| `N`         | 编码器/解码器层数  | 堆叠多少层               |
| `d_model`   | 模型维度           | 所有子层和嵌入的维度     |
| `d_ff`      | 前馈网络内部维度   | 前馈网络的隐藏层维度     |
| `h`         | 注意力头数         | 多头注意力的头数         |
| `dropout`   | Dropout 概率       | 防止过拟合               |

---

## 4.6 常用API汇总

### 4.6.1 `torch.nn.Transformer`

PyTorch 提供了内置的 Transformer 模块 `torch.nn.Transformer`，可以直接使用。

**常用参数表**：

| 参数名               | 含义         | 作用                                          |
| -------------------- | ------------ | --------------------------------------------- |
| `d_model`            | 模型维度     | 输入输出的特征维度                            |
| `nhead`              | 注意力头数   | 多头注意力的头数                              |
| `num_encoder_layers` | 编码器层数   | 编码器堆叠层数                                |
| `num_decoder_layers` | 解码器层数   | 解码器堆叠层数                                |
| `dim_feedforward`    | 前馈网络维度 | 前馈网络的隐藏层维度                          |
| `dropout`            | Dropout 概率 | 默认 0.1                                      |
| `activation`         | 激活函数     | 默认 relu，可选 gelu                          |
| `batch_first`        | 批次优先     | 若为 True，输入形状为 `(batch, seq, feature)` |

**示例**：

```python
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
src = torch.randn(10, 32, 512)  # (seq_len, batch, d_model)
tgt = torch.randn(20, 32, 512)
out = transformer(src, tgt)
print(out.shape)  # (20, 32, 512)
```

### 4.6.2 `torch.nn.MultiheadAttention`

**常用参数表**：

| 参数名        | 含义         | 作用                 |
| ------------- | ------------ | -------------------- |
| `embed_dim`   | 模型维度     | 输入特征维度         |
| `num_heads`   | 注意力头数   | 多头数量             |
| `dropout`     | Dropout 概率 | 注意力 dropout       |
| `batch_first` | 批次优先     | 输入形状是否批次在前 |

**示例**：

```python
mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
query = torch.randn(32, 10, 512)
key = torch.randn(32, 20, 512)
value = torch.randn(32, 20, 512)
attn_output, attn_weights = mha(query, key, value)
print(attn_output.shape)  # (32, 10, 512)
```

### 4.6.3 `torch.nn.TransformerEncoderLayer` 与 `torch.nn.TransformerDecoderLayer`

**TransformerEncoderLayer 常用参数表**：

| 参数名            | 含义         | 作用           |
| ----------------- | ------------ | -------------- |
| `d_model`         | 模型维度     | 输入特征维度   |
| `nhead`           | 注意力头数   | 多头数量       |
| `dim_feedforward` | 前馈网络维度 | 前馈隐藏层维度 |
| `dropout`         | Dropout 概率 | 默认 0.1       |
| `activation`      | 激活函数     | relu 或 gelu   |
| `batch_first`     | 批次优先     | 输入形状       |

**TransformerDecoderLayer** 参数类似，额外支持交叉注意力。

**示例**：

```python
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.randn(32, 10, 512)
out = transformer_encoder(src)
print(out.shape)  # (32, 10, 512)
```

---

## 4.7 总结

Transformer 是现代 NLP 的基石，其核心组件包括：

- **输入部分**：文本嵌入 + 位置编码。
- **编码器**：多头自注意力 + 前馈网络，配有残差连接和层归一化。
- **解码器**：掩码自注意力 + 交叉注意力 + 前馈网络。
- **输出层**：线性层 + softmax，输出目标词汇表概率分布。

通过堆叠多层编码器和解码器，Transformer 能够捕捉长距离依赖并支持并行计算。其后续发展出 BERT、GPT、T5 等预训练模型，推动了 NLP 领域的革命性进步。
























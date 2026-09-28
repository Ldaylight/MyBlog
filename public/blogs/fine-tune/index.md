# 一、大模型微调概述

## 1.1 什么是大模型微调

### 1.1.1 定义

大模型微调（Fine-tuning）是指在预训练大语言模型（LLM）的基础上，使用特定任务或领域的数据对模型参数进行进一步训练，使模型适应特定需求的过程。

从数学角度理解：假设预训练模型的参数为 $\theta_{\text{pre}}$，微调的目标是在给定任务数据集 $\mathcal{D}_{\text{task}} = \{(x_i, y_i)\}_{i=1}^N$ 上，找到一组新的参数 $\theta^*$，使得模型在该任务上的损失最小：

$$
\theta^* = \arg\min_{\theta} \ \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{task}}} \left[ \mathcal{L}(f_\theta(x), y) \right]
$$

其中：

- $f_\theta$：参数为 $\theta$ 的语言模型
- $\mathcal{L}$：任务损失函数（如交叉熵损失）
- $\theta$ 初始化为 $\theta_{\text{pre}}$

**关键理解**：微调不是从零开始训练，而是在已有知识的基础上"精修"。预训练模型已经学会了语言的基本规律（语法、语义、常识），微调只是引导它将这些能力应用到特定场景。

### 1.1.2 微调在LLM训练流程中的位置

大语言模型的完整训练流程通常包括三个阶段：

**（1）预训练（Pre-Training, PT）**

- **数据规模**：数万亿 Token 的无标注文本（网页、书籍、代码等）
- **目标**：学习语言的通用规律，掌握语法、语义、常识和基础推理能力
- **产物**：基础模型（Base Model），如 LLaMA-3-8B-Base
- **特点**：计算成本极高（数百万美元级别），但一次训练可反复使用

**（2）监督微调（Supervised Fine-Tuning, SFT）**

- **数据规模**：数千到数十万条高质量标注数据
- **目标**：让模型学会理解和遵循指令，按期望的格式输出
- **产物**：指令微调模型（Instruct Model），如 LLaMA-3-8B-Instruct
- **特点**：计算成本相对较低，但数据质量至关重要

**（3）偏好优化（Preference Optimization, PO）**

- **数据规模**：数万到数十万条偏好对比数据
- **目标**：让模型的输出更符合人类偏好和价值观
- **产物**：对齐模型（Aligned Model），如 ChatGPT
- **方法**：RLHF、DPO、GRPO 等

**完整流程图示**：

```
海量无标注文本
    ↓ 预训练（Pre-Training）
基础模型（Base Model）
    ↓ 监督微调（SFT）
指令微调模型（Instruct Model）
    ↓ 偏好优化（RLHF / DPO / GRPO）
对齐模型（Aligned Model）
    ↓
部署使用
```

### 1.1.3 为什么需要微调

预训练模型虽然强大，但在实际应用中面临以下局限：

**（1）领域知识不足**

通用模型缺乏特定领域的专业术语和知识。例如，医疗领域的模型需要理解"心肌梗死"、"冠状动脉"等专业术语，通用模型可能无法准确使用。

**（2）输出格式不匹配**

业务场景往往要求特定的输出格式。例如：

- 客服系统要求以友好、专业的口吻回答
- 数据分析系统要求输出 JSON 格式的结构化数据
- 法律文书要求使用规范的法律语言

通用模型的输出风格可能与业务需求不符。

**（3）指令遵循能力有限**

基础模型（Base Model）只能做"文本续写"，无法理解"请帮我总结以下内容"这样的指令。SFT 让模型学会遵循指令。

**（4）安全与价值观对齐**

需要引导模型拒绝有害请求、避免生成偏见内容、遵循特定价值观。

**（5）推理效率优化**

通过微调，可以将某些推理能力"内化"到模型参数中，减少推理时的提示词长度和计算开销。

### 1.1.4 微调的核心价值

| 价值维度 |                   具体体现                    |
| :------: | :-------------------------------------------: |
| 领域适配 |        让模型掌握特定领域的知识和术语         |
| 风格定制 |      让模型输出符合业务要求的风格和格式       |
| 能力增强 |          提升模型在特定任务上的表现           |
| 成本优化 | 通过微调减少推理时的提示词长度，降低 API 成本 |
| 数据安全 |  私有数据微调后模型可本地部署，避免数据外泄   |
| 响应速度 |   本地部署微调模型，延迟远低于调用云端 API    |

## 1.2 微调 vs 提示工程 vs RAG

### 1.2.1 三种技术路径的定位

在大模型应用中，让模型完成特定任务有三条主要技术路径，它们各有优劣，适用于不同场景。

**（1）提示工程（Prompt Engineering）**

- **核心思想**：不改变模型参数，通过精心设计提示词引导模型输出
- **实现方式**：Zero-shot、Few-shot、CoT（思维链）等
- **优点**：零成本、快速迭代、无需训练
- **缺点**：受限于模型本身能力，提示词长度有限，对复杂任务效果有限

**（2）检索增强生成（RAG）**

- **核心思想**：在生成前从外部知识库检索相关信息，注入提示词
- **实现方式**：文档向量化 + 相似度检索 + 上下文增强
- **优点**：知识可实时更新，减少幻觉，可溯源
- **缺点**：依赖检索质量，增加推理延迟，无法改变模型风格

**（3）微调（Fine-tuning）**

- **核心思想**：更新模型参数，让模型"记住"新知识或新风格
- **实现方式**：SFT、LoRA、QLoRA、DPO 等
- **优点**：性能提升显著，可深度定制，推理无额外开销
- **缺点**：需要标注数据、计算资源、训练时间

**三者对比表**：

|     对比维度     |      提示工程      |      RAG       |          微调           |
| :--------------: | :----------------: | :------------: | :---------------------: |
| 是否改变模型参数 |         否         |       否       |           是            |
|     数据需求     |      无需数据      |   知识库文档   |        标注数据         |
|     计算成本     |        极低        |      中等      |          较高           |
|     训练时间     |         无         |       无       |      数小时到数天       |
|     知识更新     |        即时        |      即时      |       需重新训练        |
|   风格改变能力   |         弱         |       无       |           强            |
|     幻觉控制     |         弱         |       强       |          中等           |
|     推理延迟     |         低         |      中等      |           低            |
|     可解释性     |        中等        |       强       |           弱            |
|     适用场景     | 通用任务、快速验证 | 知识密集型问答 | 风格/格式适配、领域专精 |

### 1.2.2 何时选择微调

微调优于提示工程和 RAG 的场景：

**（1）需要模型改变输出风格或格式**

例如，让模型以特定行业的报告格式输出，或使用特定的品牌语气。这类需求很难通过提示词稳定实现，但微调可以做到。

**（2）需要模型掌握领域特定的推理模式**

例如，医疗诊断推理、法律条文分析、金融风险评估等。这类推理模式需要模型"内化"领域知识，提示工程难以达到。

**（3）需要显著降低推理成本**

通过微调将知识内化到模型参数中，可以大幅缩短推理时的提示词长度，降低每次调用的 Token 消耗和 API 费用。

**（4）对延迟极度敏感**

RAG 需要额外的检索步骤（可能几百毫秒到几秒），而微调后的模型可以直接生成，延迟更低。

**（5）需要离线部署**

微调后的模型可以完全离线运行，适合保密要求高的场景。

### 1.2.3 三者的组合使用

实际生产中，三者往往**组合使用**而非互斥：

```
提示工程（基础） + RAG（知识） + 微调（风格/格式）
```

例如一个企业级智能客服系统可能：

1. 用**微调**让模型学会客服的语气和回答格式
2. 用**RAG**让模型能够查询最新的产品信息
3. 用**提示工程**在每次对话中引导模型聚焦当前问题

## 1.3 微调的核心决策流程

### 1.3.1 决策因素

选择微调方法时需要综合考虑以下因素：

**（1）数据规模**

- 数据量决定了微调方法的上限。数据太少时，全参微调容易过拟合。
- 一般经验：少于 1000 条数据时，优先考虑 PEFT；超过 10 万条时，可以考虑全参微调。

**（2）计算资源**

| 资源等级 |       GPU 配置        |         可用的微调方法          |
| :------: | :-------------------: | :-----------------------------: |
|  消费级  | RTX 3090/4090（24GB） |      QLoRA（7B-13B 模型）       |
|  专业级  |       A100 40GB       |  LoRA（7B-13B），QLoRA（70B）   |
|  企业级  |  A100/H100 80GB × 8   | 全参微调（7B-13B），LoRA（70B） |
| 大型企业 |      H100 × 32+       |        全参微调（70B+）         |

**（3）性能要求**

- 对性能极致追求：全参微调
- 性能与成本平衡：LoRA
- 资源极度受限：QLoRA + 小模型

**（4）部署约束**

- 推理延迟敏感：LoRA（可合并权重，无额外延迟）
- 显存受限：量化部署（GPTQ、AWQ）
- 多任务切换：LoRA 适配器可动态加载

### 1.3.2 方法选择的基本逻辑

```
开始
  │
  ├── 数据量 > 10万条 + 充足GPU（A100×8+）？
  │   ├── 是 → 全参微调（FFT）
  │   └── 否 → 继续判断
  │
  ├── 显存是否充足（>40GB）？
  │   ├── 是 → LoRA
  │   └── 否 → QLoRA
  │
  ├── 是否需要偏好对齐？
  │   ├── 是 → SFT + DPO/GRPO
  │   └── 否 → 仅 SFT
  │
  └── 是否对推理延迟敏感？
      ├── 是 → LoRA（可合并）
      └── 否 → Adapter / Prefix Tuning
```

### 1.3.3 微调的成本估算

**（1）计算成本**

以 LLaMA-3-8B 为例，不同方法的显存需求：

|   方法   | 显存需求 | 训练时间（单卡 A100） |
| :------: | :------: | :-------------------: |
| 全参微调 |  ~120GB  |       10+ 小时        |
|   LoRA   |  ~20GB   |       2-3 小时        |
|  QLoRA   |  ~10GB   |       3-4 小时        |

**（2）数据成本**

- 人工标注：每条约 1-10 元（视复杂度而定）
- 模型合成：使用 GPT-4 生成，每条约 0.1-1 元
- 数据审核：每条约 0.5-5 元

**（3）部署成本**

- 云端部署：按 GPU 小时计费
- 本地部署：一次性硬件投入

### 1.3.4 微调成功的判断标准

微调是否成功，需要从以下维度判断：

1. **任务指标**：在测试集上的自动评估指标是否达到预期
2. **人工评估**：人工抽查输出质量是否可接受
3. **泛化能力**：在未见过的相似输入上表现是否稳定
4. **遗忘程度**：模型是否丢失了预训练阶段的其他能力
5. **推理成本**：部署后的推理延迟和成本是否可接受

# 二、微调方法分类体系

## 2.1 按参数更新范围分类

### 2.1.1 全参微调（Full Fine-Tuning, FFT）

**定义**

全参微调是指更新模型的所有参数。训练时，模型的所有权重都会根据任务数据进行梯度更新。

**数学表达**

设预训练模型参数为 $\theta_{\text{pre}}$，任务损失函数为 $\mathcal{L}$，则全参微调的优化目标为：

$$
\theta^* = \arg\min_{\theta} \ \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{task}}} \left[ \mathcal{L}(f_\theta(x), y) \right]
$$

其中 $\theta$ 包含模型的所有参数，无任何冻结。

**优点**

- 理论上能达到最佳性能，模型有最大的适应空间
- 无需额外结构，实现最简单
- 微调后的模型可以直接部署，无额外推理开销

**缺点**

- 计算成本极高：需要存储所有参数的梯度、优化器状态
- 显存需求大：一个 70B 模型的全参微调需要约 1TB 显存
- 容易过拟合：小数据集上容易过拟合
- 灾难性遗忘：容易遗忘预训练阶段学到的通用能力

**显存需求估算**

全参微调的显存需求主要包括：

- 模型参数：$P$ 个参数 × 精度（FP16 = 2字节）
- 梯度：$P$ 个参数 × 精度
- 优化器状态：Adam 需要 2 倍参数量的状态（FP32）
- 激活值：与批次大小和序列长度相关

以 LLaMA-3-8B 为例（FP16 训练，Adam 优化器）：

$$
\text{显存} \approx 8\text{B} \times 2 + 8\text{B} \times 2 + 8\text{B} \times 8 + \text{激活值}
$$

$$
\approx 16\text{GB} + 16\text{GB} + 64\text{GB} + \text{激活值}
$$

$$
\approx 96\text{GB} + \text{激活值}
$$

**适用场景**

- 数据充足（>10 万条高质量标注数据）
- 计算资源丰富（多卡 A100/H100）
- 对性能要求极高
- 需要彻底改变模型行为

**典型模型参数规模下的显存需求**

| 模型规模 | 全参微调显存需求 | 最低 GPU 配置  |
| :------: | :--------------: | :------------: |
|    1B    |      ~20GB       | 单卡 A100 40GB |
|    7B    |      ~120GB      |  4× A100 40GB  |
|   13B    |      ~220GB      |  8× A100 40GB  |
|   70B    |      ~1.2TB      | 16× A100 80GB  |
|   405B   |      ~6.5TB      | 64× H100 80GB  |

### 2.1.2 参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）

**定义**

参数高效微调是指冻结预训练模型的大部分参数，仅更新少量新增参数或选定参数子集的方法。其核心思想是：预训练模型已经学到了丰富的通用知识，微调只需要调整很小一部分参数来适应新任务。

**数学表达**

设原始参数为 $\theta_{\text{pre}}$，新增的可训练参数为 $\phi$，则微调后的模型参数为：

$$
\theta^* = \theta_{\text{pre}} + \Delta\theta(\phi)
$$

其中 $\Delta\theta(\phi)$ 是由少量参数 $\phi$ 生成的参数更新量。

**优点**

- **大幅降低计算成本**：可训练参数量通常仅占全模型的 0.01%-5%
- **显存需求低**：优化器状态只需覆盖少量参数
- **减少灾难性遗忘**：大部分参数保持冻结，通用能力得以保留
- **易于部署**：适配器可以单独存储和加载，方便多任务切换
- **性能可与全参微调相当**：在许多任务上达到接近全参微调的效果

**缺点**

- 低秩约束限制了表达能力
- 某些 PEFT 方法在推理时引入额外延迟
- 对超参数（如秩）敏感
- 在 decoder-only 模型上表现可能不如全参微调

**PEFT 的核心优势**

以 LLaMA-3-8B 为例，对比全参微调和 LoRA 的资源需求：

|    指标    |     全参微调     |       LoRA       |
| :--------: | :--------------: | :--------------: |
| 可训练参数 |      80 亿       | 400 万（0.05%）  |
|  显存需求  |      ~120GB      |      ~20GB       |
|  训练时间  |     10+ 小时     |     2-3 小时     |
|  存储需求  | 16GB（完整模型） |  8MB（适配器）   |
|  推理延迟  |      无额外      | 无额外（可合并） |

**适用场景**

- 资源受限（单卡或少卡）
- 数据有限（<1 万条）
- 需要快速迭代实验
- 多任务场景（一个基础模型 + 多个适配器）
- 移动端/边缘设备部署

## 2.2 PEFT方法的五大类别

根据最新的 PEFT 综述（IEEE 2026），PEFT 方法可系统分类为五大类：

### 2.2.1 加性方法（Additive）

**定义**

在原始模型上添加可训练的模块或参数，训练时仅更新新增模块，冻结原始参数。

**核心思想**

$$
h = f_{\text{original}}(x; \theta_{\text{pre}}) + g_{\text{new}}(x; \phi)
$$

其中 $g_{\text{new}}$ 是新增的可训练模块，$\phi$ 是新增参数。

**代表方法**

- **Adapter**：在 Transformer 层中插入小型全连接网络
- **Prefix Tuning**：在每层输入前添加可训练的前缀向量
- **Prompt Tuning**：仅在输入层添加可训练的软提示向量

**特点**

- 参数量相对较大（Adapter 约 1%-5%）
- 推理时可能引入额外延迟（Adapter）
- 实现简单，容易理解

### 2.2.2 选择性方法（Selective）

**定义**

仅更新预训练参数的一个子集，其余参数保持冻结。

**核心思想**

$$
\theta^* = \theta_{\text{pre}} \odot M + \Delta\theta \odot (1 - M)
$$

其中 $M$ 是掩码矩阵，$M_{ij} = 1$ 表示该参数冻结，$M_{ij} = 0$ 表示该参数可训练。

**代表方法**

- **BitFit**：仅更新偏置项（bias）
- **Layer-wise 微调**：仅更新特定层

**特点**

- 参数量极小（BitFit 约 0.1%）
- 无需额外结构，实现极简
- 推理时无额外延迟
- 在小样本上表现良好

### 2.2.3 重参数化方法（Reparameterized）

**定义**

通过低秩分解等方式重新参数化权重更新，将高维的权重更新分解为低维表示。

**核心思想**

$$
\Delta W = B \cdot A
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**代表方法**

- **LoRA**：低秩适应
- **QLoRA**：量化 + LoRA
- **DoRA**：权重分解 + LoRA

**特点**

- 参数量极小（0.01%-1%）
- 推理时可合并，无额外延迟
- 当前最主流的 PEFT 方法

### 2.2.4 混合方法（Hybrid）

**定义**

动态组合多种 PEFT 方法，根据任务或数据特点选择最合适的方法。

**核心思想**

$$
\Delta\theta = \alpha_1 \cdot \Delta\theta_{\text{LoRA}} + \alpha_2 \cdot \Delta\theta_{\text{Adapter}} + \dots
$$

其中 $\alpha_i$ 是动态权重。

**代表方法**

- **AdaLoRA**：动态调整 LoRA 的秩分配
- **Mixed PEFT**：组合多种 PEFT 方法

**特点**

- 灵活性强，能适应不同任务
- 实现复杂度较高
- 超参数较多

### 2.2.5 统一方法（Unified）

**定义**

将多种 PEFT 思想统一到一个框架中，探索 PEFT 的本质。

**核心思想**

从理论层面统一理解各种 PEFT 方法，如将 LoRA、Adapter、Prefix Tuning 都视为某种形式的参数子空间投影。

**代表方法**

- **Unified PEFT Framework**：统一的参数化视角
- **理论分析工作**：如 LoRA 的表达能力分析

**特点**

- 理论价值大于实用价值
- 为设计新的 PEFT 方法提供理论指导
- 有助于理解 PEFT 的工作机制

## 2.3 PEFT方法选择决策表

|         场景         |     推荐方法      |          理由           |
| :------------------: | :---------------: | :---------------------: |
| 通用场景，无特殊要求 |       LoRA        |   综合最优，应用最广    |
|     显存极度受限     |       QLoRA       | 4bit 量化，显存需求最低 |
|  追求低秩下的高精度  |       DoRA        |  权重分解提升表达能力   |
|       NLU 任务       |      Adapter      |        经典有效         |
|      生成式任务      |   Prefix Tuning   |   对生成过程控制力强    |
|       极致效率       |   IA³ / BitFit    |       参数量极低        |
| 小样本 encoder 模型  |      BitFit       |  匹配甚至超越全参微调   |
|    多任务动态切换    | LoRA + 适配器加载 |   存储高效，切换灵活    |
|    需要动态秩分配    |      AdaLoRA      |    自动优化秩的分配     |
|  大模型 + 简单任务   |   Prompt Tuning   |  参数量极低，效果足够   |

## 2.4 PEFT 方法的表达能力分析

### 2.4.1 全参微调 vs PEFT 的理论差距

从理论上讲，PEFT 方法是全参微调的一个子集。全参微调可以更新任意方向的参数，而 PEFT 只能在受限的参数空间中搜索。

**LoRA 的秩约束**：

$$
\Delta W = BA, \quad \text{rank}(\Delta W) \leq r
$$

当 $r$ 足够大时，LoRA 可以逼近全参微调的效果。但实践中，$r$ 通常取 8-64，远小于 $\min(d, k)$，因此 LoRA 的解空间是全参微调解空间的一个严格子集。

### 2.4.2 为什么 PEFT 仍然有效

尽管表达能力受限，PEFT 仍然在实践中表现良好，原因包括：

1. **预训练模型已经学到了丰富的表示**：微调只需要在已有基础上做小幅调整。
2. **任务的本质是"低秩"的**：许多下游任务只需要改变模型的输出分布，而不需要重新学习所有表示。
3. **正则化效果**：PEFT 的约束反而起到了防止过拟合的作用。

### 2.4.3 关键实验发现

最新的样本效率缩放定律研究（TACL 2026）揭示了架构依赖的样本效率差异：

- **Encoder-only 模型**（如 BERT）：多种 PEFT 方法在约 700 条样本以下能匹配或超越全参微调。
- **Decoder-only 模型**（如 LLaMA）：全参微调在所有观测样本量下都优于 LoRA。

**结论**：对于 decoder-only 的生成式大模型，PEFT 的样本效率优势并不明显。选择 PEFT 的主要理由是**资源约束**而非样本效率。

## 2.5 总结

微调方法的选择是一个多维度权衡的过程。全参微调提供最大性能，但代价高昂；PEFT 通过参数高效的设计，在资源受限的场景下提供了接近全参微调的性能。理解 PEFT 方法的五大类别（加性、选择性、重参数化、混合、统一）及其代表方法，是选择合适微调策略的基础。

对于大多数实践者而言，**LoRA 和 QLoRA 是默认的起点**——它们在资源效率、性能和易用性之间取得了最佳平衡。当有明确的需求（如极低秩下的高精度、极致效率、动态秩分配）时，再考虑 DoRA、IA³、AdaLoRA 等专门方法。

# 三、参数高效微调技术详解——更新标记

## 3.1 LoRA（Low-Rank Adaptation）

### 3.1.1 核心原理

LoRA 由 Hu 等人于 2021 年提出（ICLR 2022），是当前应用最广泛的参数高效微调方法。其核心思想是：在原始权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 旁边新增一条旁路，由两个低秩矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 相乘而成，其中 $r \ll \min(d, k)$。

![image-20260928092047934](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260928092047934.png)

前向传播时，输入同时通过原始权重和 LoRA 旁路，输出相加：

$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

训练时，$W_0$ 被冻结，仅更新 $B$ 和 $A$。为了控制 LoRA 更新的强度，引入缩放因子 $\alpha$，实际更新量为：

$$
\Delta W = \frac{\alpha}{r} \cdot B A
$$

其中 $\alpha$ 是缩放因子，$r$ 是秩。$\alpha/r$ 的比例决定了 LoRA 更新的幅度。

**参数量的减少非常显著**。以一个 $(4096, 4096)$ 的权重矩阵为例，原始参数量是 1670 万。换成 rank=8 的 LoRA 之后，$A$ 的维度 $(8, 4096)$ 有 32,768 个参数，$B$ 的维度 $(4096, 8)$ 同样 32,768 个，加起来总共 65,536——比原来少了 99.6%。

LoRA 论文中的一个关键发现是：微调过程中大部分有意义的权重更新本来就集中在低维子空间里，所以把更新约束在低秩矩阵上并不会造成多大损失。

![image-20260928093823621](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260928093823621.png)

### 3.1.2 数学推导

#### **3.1.2.1 低秩假设的来源**——奇异值分解

微调过程中，权重更新矩阵 $\Delta W$ 的秩通常远小于 $\min(d, k)$。Aghajanyan 等人（2020）的研究表明，预训练语言模型在适应下游任务时，其权重更新具有极低的内在维度。LoRA 正是基于这一观察，将 $\Delta W$ 参数化为两个低秩矩阵的乘积：

$$
\Delta W = B A
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

为什么可以用两个小矩阵 B 和 A 就能表示一个足够大的更新量？

线性代数中有一个基本事实：

如果一个矩阵 $\Delta W$ 的秩不超过 r，那么它一定能写成两个矩阵的乘积：
$$
\Delta W = BA
$$
其中 $B \in \mathbb{R}^{m \times r}$，$A \in \mathbb{R}^{r \times n}$

这个事实是**严格的，不需要 SVD 也能证明**：  

因为 $\Delta W$ 的秩是 r，所以它的列空间维数是 r。取列空间的一组基构成 r，那么$\Delta W$ 的每一列都可以表示为 B 的列的线性组合，这些系数就构成 A。所以：秩为 r 的矩阵，一定可以分解成 $m \times r$ 和 $r \times n$ 两个矩阵的乘积。

参数量从：$m \times n$，降到：$m \times r + r \times n = r(m+n)$，当 $r \ll \min(m,n) $时，参数量大幅减少。



SVD 进一步告诉你：任何矩阵 $W$ 都可以写成 $U \Sigma V^T$，其中奇异值从大到小排列。

如果只保留前 $r$ 个奇异值，就得到：
$$
W \approx U_r \Sigma_r V_r^T
$$
这就是 W 的秩 r 最优近似。Eckart-Young 定理保证：在所有秩不超过 r 的矩阵中，这个近似在 Frobenius 范数下误差最小。

把 $U_r \Sigma_r$合并成 B，把 $V_r^T$ 看作 A，就得到：
$$
W \approx BA
$$
所以 SVD 解释的是：

> 如果原矩阵可以用低秩近似，那么它就能拆成两个小矩阵；而且这种拆分在重构误差意义下是最优的。

LoRA 把这个逻辑反过来用：

> 不是对 \(W_0\) 做 SVD，而是假设更新量 \(\Delta W\) 是低秩的，然后用 \(BA\) 来参数化它，让训练去学 \(A\) 和 \(B\)。

---

**为什么可以假设$\Delta W$ 是低秩的？**

- **数学上**：低秩矩阵一定能拆成两个小矩阵，这是确定的。
- **经验上**：微调时的 $\Delta W$ 是否真的是低秩的？这是 LoRA 的假设。

研究者发现：

1. 预训练权重本身往往有冗余，奇异值衰减很快；
2. 下游任务适配时，模型不需要改变所有方向，只需要调整一个低维子空间；
3. 实验表明，r 取 4、8、16 就能达到接近全参微调的效果。

所以 LoRA 的合理性来自两部分：

- **SVD/低秩分解**：提供“低秩可以拆成两个小矩阵”的数学基础；
- **经验观察**：微调更新确实可以用低秩近似，不损失太多性能。

**（2）前向传播推导**

对于输入 $x \in \mathbb{R}^k$，原始线性层输出为：

$$
h_{\text{orig}} = W_0 x
$$

LoRA 旁路输出为：

$$
h_{\text{lora}} = B A x
$$

最终输出为两者之和：

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

**（3）初始化策略**

为了在训练开始时保持模型行为与预训练一致，LoRA 采用以下初始化：

- $A$ 使用随机高斯初始化（均值为 0，方差为 $1/r$）
- $B$ 初始化为全零矩阵

因此，训练开始时 $B A = 0$，$\Delta W = 0$，模型输出与预训练模型完全一致。随着训练进行，$B$ 和 $A$ 逐渐更新，模型逐步适应新任务。

**（4）参数量分析**

原始权重参数量：$d \times k$

LoRA 参数量：$d \times r + r \times k = r(d + k)$

参数减少比例：

$$
\frac{r(d+k)}{d \times k} = r \left( \frac{1}{k} + \frac{1}{d} \right)
$$

当 $d = k = 4096$，$r = 8$ 时：

$$
\frac{8 \times (4096 + 4096)}{4096 \times 4096} = \frac{65536}{16777216} \approx 0.0039 = 0.39\%
$$

即参数量仅为原来的 0.39%，减少了 99.61%。

**（5）缩放因子的作用**

缩放因子 $\alpha/r$ 的作用是：当调整秩 $r$ 时，保持 LoRA 更新的有效幅度大致不变。这是因为 $B A$ 的方差会随 $r$ 增大而增大（因为 $A$ 的每个元素方差为 $1/r$，$B$ 初始为 0，但训练后 $B$ 和 $A$ 的乘积的尺度会受 $r$ 影响）。除以 $r$ 可以稳定不同秩下的训练动态。

### 3.1.3 关键参数

|       参数       |       含义       |     推荐值     |              说明              |
| :--------------: | :--------------: | :------------: | :----------------------------: |
|    `r`（秩）     |   低秩矩阵的秩   |      8-64      | 越大表达能力越强，但参数量增加 |
|   `lora_alpha`   |     缩放因子     |     16-64      |       通常设为 r 的 2 倍       |
| `target_modules` | 应用 LoRA 的模块 | q_proj, v_proj |       可扩展到所有线性层       |
|  `lora_dropout`  |   Dropout 概率   |    0.05-0.1    |           防止过拟合           |
|      `bias`      |   偏置处理方式   |     "none"     |         通常不训练偏置         |

### 3.1.4 代码实现

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha

        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        # LoRA 旁路
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x):
        # 原始权重输出
        original = nn.functional.linear(x, self.weight)
        # LoRA 旁路输出
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return original + lora_out * self.scaling

    def merge(self):
        """将 LoRA 权重合并回原始权重，推理时无额外延迟"""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.r = 0
```

使用 Hugging Face PEFT 库实现 LoRA：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# 输出: trainable params: 6,815,744 || all params: 8,037,404,672 || trainable%: 0.0848
```

### 3.1.5 优势与局限

**优势**：

- 参数量仅占全模型的 0.01%-1%
- 推理时无额外延迟（可将 LoRA 权重合并回原模型）
- 通用性强，是最常用的默认选择
- 论文已验证：微调过程中有意义的权重更新集中在低维子空间

**局限**：

- 低秩约束限制了表达能力，其解空间是全参微调解空间的严格子集
- 对超参数（尤其是秩的选择）敏感
- 在 decoder-only 模型上，样本效率可能不如全参微调

## 3.2 QLoRA（Quantized LoRA）

### 3.2.1 核心原理

QLoRA 由 Dettmers 等人于 2023 年提出，在 LoRA 的基础上引入了模型量化。其核心策略是：加载时将预训练模型量化为 4bit 精度，计算时反量化为 16bit。QLoRA 集成了三个关键组件：

**（1）NF4（NormalFloat 4-bit）**

NF4 是一种信息论最优的量化格式，专门为正态分布的权重设计。它根据正态分布的 CDF 来划分量化级别——在权重密集的区域（0 附近）分配更多的量化级别，整体误差更小。相比之下，标准的 Int4 量化像“均匀刻度的尺子”，而 NF4 像“对数刻度的尺子”。

**（2）双重量化（Double Quantization）**

对量化常数本身再做一次量化。传统的量化需要为每个块存储一个缩放因子（scale），双重量化将这些缩放因子也进行量化，进一步减少显存占用约 0.37 bits/参数。

**（3）分页优化器（Paged Optimizers）**

利用 NVIDIA 统一内存，在 GPU 显存不足时自动将优化器状态分页到 CPU 内存，避免 OOM 错误。

QLoRA 的前向传播表达式为：

$$
Y^{\mathrm{BF16}} = X^{\mathrm{BF16}} \cdot \mathrm{doubleDequant}(c_1^{\mathrm{FP32}}, c_2^{\mathrm{k-bit}}, W^{\mathrm{NF4}}) + X^{\mathrm{BF16}} L_1^{\mathrm{BF16}} L_2^{\mathrm{BF16}}
$$

其中基础模型权重被量化为 NF4 精度，而 LoRA 适配器 $(L_1, L_2)$ 在 16bit 精度下运行。

**显存节省效果**：QLoRA 使得在单张 48GB GPU 上微调 65B 参数模型成为可能，同时保持完整的 16bit 性能。实践中，QLoRA 仅需全模型 0.1% 的可训练参数即可达到与全参微调相当的结果。

### 3.2.2 量化推导

**（1）标准线性量化**

将浮点权重 $W$ 量化为 $k$ 位整数：

$$
W \approx s \cdot q
$$

其中 $s$ 是缩放因子（scale），$q$ 是量化后的整数。反量化时：

$$
\hat{W} = s \cdot q
$$

量化误差为：

$$
\epsilon = W - \hat{W}
$$

**（2）分块量化**

为了减少量化误差，将权重分成多个块，每个块使用独立的缩放因子：

$$
W_i \approx s_i \cdot q_i, \quad i = 1, 2, \dots, B
$$

其中 $B$ 是块的数量。块内共享缩放因子，块间独立。

**（3）NF4 量化**

NF4 的核心是：对于正态分布的权重，最优的量化级别不是均匀分布的，而是按照正态分布的分位数来划分。具体地，NF4 将 $[-1, 1]$ 区间划分为 $2^4 = 16$ 个级别，每个级别的边界由标准正态分布的分位数确定：

$$
q_i = \Phi^{-1}\left( \frac{i + 0.5}{16} \right), \quad i = 0, 1, \dots, 15
$$

其中 $\Phi^{-1}$ 是标准正态分布的逆累积分布函数。这样，在权重密集的区域（0 附近），量化级别更密集，误差更小。

**（4）双重量化**

传统量化中，每个块需要存储一个 FP32 的缩放因子 $s_i$。双重量化对这些缩放因子再进行一次量化：

$$
s_i \approx s_{\text{outer}} \cdot q_i^{\text{inner}}
$$

其中 $s_{\text{outer}}$ 是外层缩放因子，$q_i^{\text{inner}}$ 是内层量化整数。这样，原本每个块需要一个 FP32 的缩放因子，现在只需要一个 FP32 的外层缩放因子加上少量 8bit 的内层整数，进一步节省显存。

**（5）显存节省计算**

以 65B 模型为例：

- FP16 模型权重：$65 \times 10^9 \times 2 \text{ bytes} = 130\text{GB}$
- NF4 量化后：$65 \times 10^9 \times 0.5 \text{ bytes} = 32.5\text{GB}$
- 双重量化再节省：约 $65 \times 10^9 \times 0.37 \text{ bits} \approx 3\text{GB}$
- 最终显存：约 29.5GB，单张 48GB GPU 即可加载

### 3.2.3 关键参数

|            参数             |      含义      | 推荐值 |          说明          |
| :-------------------------: | :------------: | :----: | :--------------------: |
|       `load_in_4bit`        | 是否 4bit 加载 |  True  |       启用 QLoRA       |
|    `bnb_4bit_quant_type`    |    量化类型    | "nf4"  | 使用 NormalFloat 4-bit |
| `bnb_4bit_use_double_quant` |    双重量化    |  True  |       进一步压缩       |
|  `bnb_4bit_compute_dtype`   |    计算精度    | "bf16" |   反量化后的计算精度   |

### 3.2.4 代码实现

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 4bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
```

### 3.2.5 显存需求对比

以 LLaMA-3-8B 为例：

|       方法       | 模型权重显存 | 优化器状态 | 总显存需求 |
| :--------------: | :----------: | :--------: | :--------: |
| 全参微调（FP16） |     16GB     |    64GB    |   ~120GB   |
|   LoRA（FP16）   |     16GB     |   0.2GB    |   ~20GB    |
|  QLoRA（4bit）   |     4GB      |   0.2GB    |   ~10GB    |

### 3.2.6 适用场景

- 显存极度受限（单张 24GB 消费级 GPU）
- 需要微调 7B-13B 参数模型
- 低数据场景下效果尤为突出
- 对未见主题的外推能力更好

## 3.3 DoRA（Weight-Decomposed Low-Rank Adaptation）

### 3.3.1 核心原理

DoRA 由 Liu 等人于 2024 年提出（ICML 2024），其核心创新在于**权重分解**。DoRA 首先将预训练权重矩阵 $W$ 分解为**幅度（magnitude）$m$** 和**方向（direction）$V$** 两个分量：

$$
W = m \frac{V}{\|V\|_c}
$$

其中 $\|\cdot\|_c$ 表示逐列的向量范数。然后，DoRA 仅对方向分量应用 LoRA：

$$
W' = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

其中 $W_0$ 是预训练权重，$BA$ 表示低秩更新。

这种分解方法的灵感来自 Weight Normalization 技术，旨在简化低秩更新的学习任务。通过分别微调幅度和方向，DoRA 增强了 LoRA 的学习能力和训练稳定性，同时**不引入任何额外的推理延迟**（与 LoRA 一样，权重可以合并）。

DoRA 在 LLaMA、LLaVA 和 VL-BART 上，在常识推理、视觉指令调优和图像/视频文本理解等多种下游任务上，一致地优于 LoRA。

### 3.3.2 数学推导

**（1）权重分解**

将权重矩阵 $W \in \mathbb{R}^{d \times k}$ 按列分解：

$$
W = m \frac{V}{\|V\|_c}
$$

其中 $m \in \mathbb{R}^{1 \times k}$ 是幅度向量（每列的范数），$V \in \mathbb{R}^{d \times k}$ 是方向矩阵。$\|V\|_c$ 表示对 $V$ 的每一列计算 L2 范数，得到 $1 \times k$ 的向量。

**（2）DoRA 的参数化**

DoRA 将 $W$ 分解为：

$$
W' = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

其中：

- $W_0$：冻结的预训练权重
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$：可训练的低秩矩阵
- $m \in \mathbb{R}^{1 \times k}$：可训练的幅度向量

**（3）训练过程**

训练时，$W_0$ 冻结，仅更新 $B$、$A$ 和 $m$。前向传播时，先计算 $W_0 + BA$，再计算其列范数，最后乘以幅度 $m$。

**（4）与 LoRA 的对比推导**

LoRA 的更新为：

$$
W_{\text{LoRA}} = W_0 + BA
$$

DoRA 的更新为：

$$
W_{\text{DoRA}} = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

可以看到，DoRA 在 LoRA 的基础上增加了一个幅度缩放因子 $m$，并对方向进行归一化。这使得 DoRA 可以独立地调整权重的幅度和方向，而 LoRA 只能同时调整两者。

**（5）参数量对比**

LoRA 参数量：$d \times r + r \times k = r(d + k)$

DoRA 参数量：$r(d + k) + k$（额外增加幅度向量 $m$ 的 $k$ 个参数）

由于 $k \ll r(d+k)$（通常 $d, k$ 很大），DoRA 的额外参数量可以忽略不计。

### 3.3.3 与 LoRA 的对比

| 维度         | LoRA         | DoRA                        |
| ------------ | ------------ | --------------------------- |
| 权重分解     | 无           | 幅度 + 方向                 |
| 低秩更新位置 | 原始权重旁路 | 仅方向分量                  |
| 表达能力     | 受低秩约束   | 更强（幅度独立学习）        |
| 推理延迟     | 无额外       | 无额外                      |
| 低秩时表现   | 一般         | 显著优于 LoRA               |
| 参数量       | 0.01%-1%     | 略多于 LoRA（增加幅度参数） |

### 3.3.4 代码实现

使用 Hugging Face PEFT 实现 DoRA：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=True  # 启用 DoRA
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
peft_model = get_peft_model(model, config)
```

### 3.3.5 适用场景

- 对低秩微调精度要求较高的场景
- 在低秩（如 r=4 或 r=8）时相比 LoRA 有明显提升
- 希望在相同参数预算下获得更好效果
- 实践中建议：在提升秩之前先尝试 DoRA

## 3.4 Adapter

### 3.4.1 核心原理

Adapter 由 Houlsby 等人于 2019 年提出，是最早的 PEFT 方法之一。其核心思想是在 Transformer 层的特定位置（通常是自注意力和前馈网络之后）插入小型全连接网络。训练时仅更新 Adapter 参数，冻结原模型参数。

Adapter 的典型结构是一个瓶颈层：

$$
h = W_{\text{up}} \cdot \text{ReLU}(W_{\text{down}} \cdot x)
$$

其中 $W_{\text{down}} \in \mathbb{R}^{r \times d}$ 将维度从 $d$ 降到 $r$，$W_{\text{up}} \in \mathbb{R}^{d \times r}$ 再升回 $d$。

### 3.4.2 数学推导

**（1）瓶颈结构**

给定输入 $x \in \mathbb{R}^d$，Adapter 首先将其降维到瓶颈维度 $r$：

$$
z = W_{\text{down}} x, \quad z \in \mathbb{R}^r
$$

然后应用非线性激活：

$$
z' = \text{ReLU}(z)
$$

最后升维回原始维度：

$$
h = W_{\text{up}} z', \quad h \in \mathbb{R}^d
$$

**（2）残差连接**

Adapter 通常带有残差连接：

$$
\text{Adapter}(x) = x + W_{\text{up}} \cdot \text{ReLU}(W_{\text{down}} \cdot x)
$$

残差连接确保当 Adapter 输出为 0 时，模块退化为恒等映射，不影响原始模型行为。

**（3）参数量分析**

Adapter 的参数量：

$$
d \times r + r \times d = 2dr
$$

当 $d = 768$，$r = 64$ 时，参数量为 $2 \times 768 \times 64 = 98,304$。相比原始层参数量 $768 \times 768 = 589,824$，Adapter 参数量约为原来的 16.7%。

**（4）插入位置**

Houlsby 等人提出在每个 Transformer 层中插入两个 Adapter：

- 第一个在自注意力子层之后
- 第二个在前馈网络子层之后

每个 Adapter 都有残差连接。

### 3.4.3 特点

- 参数量约占全模型的 1%-5%
- 推理时引入额外计算开销（约 10%-20% 延迟）
- 适合 NLU 任务
- 在 PEFT-Bench 基准中，Adapter 类方法的平均性能（$P_{avg}$）约为 74-80

### 3.4.4 代码实现

```python
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return residual + x
```

## 3.5 Prefix Tuning

### 3.5.1 核心原理

Prefix Tuning 由 Li 和 Liang 于 2021 年提出。其核心思想是在每一层的输入前添加一组可训练的“前缀”向量（连续向量）。这些前缀向量不是离散的 token，而是可训练的连续嵌入，引导模型生成期望的输出。

对于每一层 $l$，前缀向量为 $P_l \in \mathbb{R}^{p \times d}$，其中 $p$ 是前缀长度。前向传播时，前缀与输入序列拼接：

$$
\text{input}_l = [P_l; x_l]
$$

### 3.5.2 数学推导

**（1）注意力中的前缀**

在自注意力中，输入序列 $X \in \mathbb{R}^{n \times d}$ 首先通过线性变换得到 $Q, K, V$：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

加入前缀后，输入变为 $[P; X]$，其中 $P \in \mathbb{R}^{p \times d}$。则：

$$
Q' = [P; X] W_Q, \quad K' = [P; X] W_K, \quad V' = [P; X] W_V
$$

注意力计算变为：

$$
\text{Attention}(Q', K', V') = \text{softmax}\left( \frac{Q' K'^\top}{\sqrt{d_k}} \right) V'
$$

前缀向量 $P$ 作为可训练参数，通过注意力机制影响后续所有位置的表示。

**（2）参数化**

Prefix Tuning 有两种参数化方式：

- **直接参数化**：每层直接学习前缀矩阵 $P_l \in \mathbb{R}^{p \times d}$
- **重参数化**：通过一个 MLP 生成前缀：$P_l = \text{MLP}(P'_l)$，其中 $P'_l \in \mathbb{R}^{p \times d'}$ 是更小的可训练矩阵

重参数化可以稳定训练，尤其在小数据集上。

**（3）参数量分析**

直接参数化：$L \times p \times d$，其中 $L$ 是层数。

例如，$L = 12$，$p = 20$，$d = 768$，参数量为 $12 \times 20 \times 768 = 184,320$，约占全模型（110M）的 0.17%。

### 3.5.3 特点

- 参数量极小（通常低于 0.1%）
- 会占用额外的输入 token 位置，减少有效上下文长度
- 在 PEFT-Bench 中，Prefix Tuning 的平均性能较低（$P_{avg}$ 约 45.9），但参数效率极高
- 适合生成式任务

### 3.5.4 代码实现

使用 PEFT 库：

```python
from peft import PrefixTuningConfig, get_peft_model

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # 前缀长度
    prefix_projection=True
)

peft_model = get_peft_model(model, config)
```

## 3.6 Prompt Tuning

### 3.6.1 核心原理

Prompt Tuning 是 Prefix Tuning 的简化版本，由 Lester 等人于 2021 年提出。它仅在输入层添加可训练的软提示（soft prompt）向量，不修改中间层。这意味着 Prompt Tuning 只影响模型的输入嵌入，而不改变每一层的注意力计算。

### 3.6.2 数学推导

**（1）输入嵌入**

给定输入 token 序列 $x = (x_1, \dots, x_n)$，首先通过嵌入层得到：

$$
E = \text{Embedding}(x) \in \mathbb{R}^{n \times d}
$$

**（2）软提示拼接**

Prompt Tuning 添加一个可训练的软提示矩阵 $P \in \mathbb{R}^{p \times d}$，与输入嵌入拼接：

$$
E' = [P; E] \in \mathbb{R}^{(p+n) \times d}
$$

**（3）前向传播**

拼接后的嵌入 $E'$ 直接输入 Transformer 层，后续计算与标准 Transformer 完全一致。软提示 $P$ 作为可训练参数，通过梯度下降更新。

**（4）与 Prefix Tuning 的区别**

- Prefix Tuning 在每一层都添加前缀，影响每一层的注意力计算。
- Prompt Tuning 仅在输入层添加软提示，只影响输入嵌入。
- Prompt Tuning 参数量更少，但表达能力也更弱。

### 3.6.3 特点

- 参数量极低（低于 0.01%）
- 在 PEFT-Bench 中，Prompt Tuning 的平均性能最低（$P_{avg}$ 约 50.0）
- 适合大规模模型 + 简单任务
- 当模型规模增大时，Prompt Tuning 的效果会提升

## 3.7 IA³（Infused Adapter by Inhibiting and Amplifying Inner Activations）

### 3.7.1 核心原理

IA³ 由 Liu 等人于 2022 年提出。它通过学习三个缩放向量，分别对注意力机制中的 Key（$k$）、Value（$v$）和前馈网络中的激活值进行逐元素缩放。

对于 Key 的缩放向量 $l_k \in \mathbb{R}^{d_k}$：

$$
k' = l_k \odot k
$$

其中 $\odot$ 表示逐元素乘法。类似地，Value 和 FFN 也各有缩放向量。

### 3.7.2 数学推导

**（1）Key 的缩放**

在注意力中，Key 矩阵 $K \in \mathbb{R}^{n \times d_k}$ 被缩放向量 $l_k \in \mathbb{R}^{d_k}$ 逐元素缩放：

$$
K' = K \odot l_k
$$

其中 $\odot$ 表示广播逐元素乘法，即 $K'_{ij} = K_{ij} \cdot l_{k,j}$。

**（2）Value 的缩放**

类似地，Value 矩阵 $V \in \mathbb{R}^{n \times d_v}$ 被缩放：

$$
V' = V \odot l_v
$$

**（3）FFN 的缩放**

前馈网络的中间激活值被缩放：

$$
\text{FFN}(x) = W_2 \cdot (l_{\text{ff}} \odot \text{ReLU}(W_1 x))
$$

**（4）参数量分析**

IA³ 的参数量为：

$$
d_k + d_v + d_{\text{ff}}
$$

对于 $d_k = d_v = 64$，$d_{\text{ff}} = 3072$，参数量为 $64 + 64 + 3072 = 3200$，约占全模型的 0.003%。

### 3.7.3 特点

- 参数量极低（低于 0.01%）
- 推理时无额外延迟
- 支持混合任务批次（mixed-task batches）
- 在 PEFT-Bench 中，IA³ 的平均性能（$P_{avg}$ 约 74.7）与 Adapter 相当，但参数量远低于 Adapter

### 3.7.4 代码实现

```python
from peft import IA3Config, get_peft_model

config = IA3Config(
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"]
)

peft_model = get_peft_model(model, config)
```

## 3.8 BitFit

### 3.8.1 核心原理

BitFit 由 Ben Zaken 等人于 2022 年提出。它仅微调模型中的偏置项（bias），冻结所有其他参数。研究发现，在 encoder-only 模型上，当样本量低于约 700 条时，BitFit 能匹配甚至超越全参微调的表现。

### 3.8.2 数学推导

**（1）偏置项的作用**

在标准线性层中：

$$
y = W x + b
$$

其中 $W$ 是权重矩阵，$b$ 是偏置向量。BitFit 冻结 $W$，仅更新 $b$。

**（2）参数量分析**

对于一个 Transformer 模型，偏置项包括：

- 线性层的偏置：每层 $d$ 个参数
- LayerNorm 的偏置：每层 $d$ 个参数

总参数量约为：

$$
N_{\text{bias}} \approx L \times (4d + 2d) = 6Ld
$$

对于 $L = 12$，$d = 768$，参数量约为 $6 \times 12 \times 768 = 55,296$，约占全模型的 0.05%。

**（3）为什么有效**

BitFit 的有效性可以解释为：偏置项决定了激活函数的偏移量，调整偏置可以改变神经元激活的阈值，从而适应新任务。同时，偏置项数量少，不容易过拟合。

### 3.8.3 特点

- 参数量极小（约 0.1%）
- 无需额外结构，实现极简
- 推理时无额外延迟
- 在 PEFT-Bench 中，BitFit 的平均性能（$P_{avg}$ 约 75.3），性价比极高
- 适合小样本 encoder 模型

### 3.8.4 代码实现

```python
# 仅训练 bias 参数
for name, param in model.named_parameters():
    if "bias" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

## 3.9 AdaLoRA

### 3.9.1 核心原理

AdaLoRA 由 Zhang 等人于 2023 年提出。它在 LoRA 的基础上，动态调整各层低秩矩阵的秩分配。重要的层分配更大的秩，不重要的层分配更小的秩，实现更高效的参数利用。

AdaLoRA 使用奇异值分解（SVD）的形式参数化 LoRA 更新：

$$
\Delta W = P \Lambda Q
$$

其中 $\Lambda$ 是对角矩阵，对角元素为奇异值。训练过程中，AdaLoRA 根据重要性评分动态裁剪较小的奇异值，从而自适应地分配秩。

### 3.9.2 数学推导

**（1）SVD 参数化**

标准 LoRA 将更新参数化为 $BA$。AdaLoRA 将其参数化为：

$$
\Delta W = P \Lambda Q
$$

其中 $P \in \mathbb{R}^{d \times r}$ 是正交矩阵，$Q \in \mathbb{R}^{r \times k}$ 是正交矩阵，$\Lambda \in \mathbb{R}^{r \times r}$ 是对角矩阵，对角元素为奇异值 $\lambda_1, \dots, \lambda_r$。

**（2）重要性评分**

AdaLoRA 使用奇异值的绝对值作为重要性评分：

$$
I_i = |\lambda_i|
$$

在训练过程中，定期根据重要性评分裁剪掉最小的奇异值，即将其置零。

**（3）动态秩分配**

假设总参数预算为 $B$，AdaLoRA 根据各层的重要性评分动态分配秩。重要的层保留更多的奇异值，不重要的层保留更少的奇异值。

具体地，AdaLoRA 使用以下策略：

- 每 $T$ 步重新计算各层的重要性评分
- 根据评分分配各层的目标秩
- 裁剪掉低于阈值的奇异值

**（4）正交性约束**

为了确保 $P$ 和 $Q$ 的正交性，AdaLoRA 在损失函数中添加正交性正则项：

$$
\mathcal{L}_{\text{orth}} = \|P^\top P - I\|_F^2 + \|Q Q^\top - I\|_F^2
$$

其中 $\|\cdot\|_F$ 是 Frobenius 范数。

### 3.9.3 特点

- 参数量与 LoRA 相当
- 自动优化秩的分配，无需手动设置每层的秩
- 实现复杂度较高
- 在 PEFT-Bench 中表现优于固定秩的 LoRA

### 3.9.4 代码实现

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    task_type="CAUSAL_LM",
    init_r=12,          # 初始秩
    target_r=8,         # 目标秩
    tinit=200,          # 开始调整秩的步数
    tfinal=1000,        # 停止调整秩的步数
    deltaT=10,          # 调整间隔
    target_modules=["q_proj", "v_proj"]
)

peft_model = get_peft_model(model, config)
```

## 3.10 PEFT 方法全面对比

### 3.10.1 性能与效率对比

基于 PEFT-Bench 基准（2026）的实测数据：

|     方法      | 平均性能（$P_{avg}$） | 参数效率成本 | 推理开销 | 参数量占比 |
| :-----------: | :-------------------: | :----------: | :------: | :--------: |
|     LoRA      |         80.1          |     0.99     |    无    |  0.01%-1%  |
|   LNTuning    |         77.8          |     1.00     |    无    |  约 0.1%   |
|    BitFit     |         75.3          |     1.00     |    无    |  约 0.1%   |
|      IA³      |         74.7          |     1.00     |    无    |   <0.01%   |
| Prompt Tuning |         50.0          |     1.00     | +tokens  |   <0.01%   |
| Prefix Tuning |         45.9          |     0.97     | +tokens  |   <0.1%    |
|   P-Tuning    |         51.7          |     0.95     | +tokens  |   <0.1%    |

> 注：参数效率成本（$cost_p$）越高表示参数效率越好，1.00 为最优。

### 3.10.2 方法选择决策表

|         场景         |     推荐方法      |          理由           |
| :------------------: | :---------------: | :---------------------: |
| 通用场景，无特殊要求 |       LoRA        |   综合最优，应用最广    |
|     显存极度受限     |       QLoRA       | 4bit 量化，显存需求最低 |
|  追求低秩下的高精度  |       DoRA        |  权重分解提升表达能力   |
|       NLU 任务       |      Adapter      |        经典有效         |
|      生成式任务      |   Prefix Tuning   |   对生成过程控制力强    |
|       极致效率       |   IA³ / BitFit    | 参数量极低，无推理开销  |
| 小样本 encoder 模型  |      BitFit       |  匹配甚至超越全参微调   |
|    多任务动态切换    | LoRA + 适配器加载 |   存储高效，切换灵活    |
|    需要动态秩分配    |      AdaLoRA      |    自动优化秩的分配     |
|  大模型 + 简单任务   |   Prompt Tuning   |  参数量极低，效果足够   |
|  低秩时追求更高精度  |       DoRA        | 相同参数预算下通常更好  |

### 3.10.3 表达能力分析

从理论上讲，PEFT 方法是全参微调的一个子集。全参微调可以更新任意方向的参数，而 PEFT 只能在受限的参数空间中搜索。

**LoRA 的秩约束**：

$$
\Delta W = BA, \quad \text{rank}(\Delta W) \leq r
$$

当 $r$ 足够大时，LoRA 可以逼近全参微调的效果。但实践中，$r$ 通常取 8-64，远小于 $\min(d, k)$，因此 LoRA 的解空间是全参微调解空间的一个严格子集。

**为什么 PEFT 仍然有效**：

1. **预训练模型已经学到了丰富的表示**：微调只需要在已有基础上做小幅调整。
2. **任务的本质是“低秩”的**：许多下游任务只需要改变模型的输出分布。
3. **正则化效果**：PEFT 的约束反而起到了防止过拟合的作用。

**关键实验发现**（TACL 2026）：

- **Encoder-only 模型**（如 BERT）：多种 PEFT 方法在约 700 条样本以下能匹配或超越全参微调。
- **Decoder-only 模型**（如 LLaMA）：全参微调在所有观测样本量下都优于 LoRA。

**结论**：对于 decoder-only 的生成式大模型，PEFT 的样本效率优势并不明显。选择 PEFT 的主要理由是**资源约束**而非样本效率。

# 四、监督微调（SFT）

## 4.1 SFT 的定义与作用

### 4.1.1 定义

监督微调（Supervised Fine-Tuning, SFT）是使用“输入-输出”配对数据对预训练模型进行训练的过程。目标是让模型学习特定任务的输出格式、风格和领域词汇。

**直观理解**：预训练模型像是一个博览群书但不懂规矩的学者——它知道很多知识，但不知道怎么按照要求回答问题。SFT 就是给这个学者做“岗前培训”，通过大量“问题-标准答案”的示例，教会它如何恰当地回应。

**数学表达**：

设预训练模型为 $f_\theta$，SFT 数据集为 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，其中 $x_i$ 是指令/输入，$y_i$ 是期望的输出。SFT 的优化目标为：

$$
\theta^* = \arg\min_{\theta} \ \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ -\sum_{t=1}^{|y|} \log P_\theta(y_t \mid y_{<t}, x) \right]
$$

其中 $P_\theta(y_t \mid y_{<t}, x)$ 是模型在给定输入 $x$ 和之前输出 $y_{<t}$ 的条件下，生成当前 token $y_t$ 的概率。这是一个标准的自回归语言建模损失。

### 4.1.2 SFT 在微调流程中的位置

SFT 是微调流程中最基础、最常用的环节：

```
预训练基础模型（Base Model）
    ↓ SFT（监督微调）
指令微调模型（Instruct Model）
    ↓ 偏好优化（DPO/GRPO/RLHF）
对齐模型（Aligned Model）
```

SFT 让模型从“文本续写器”变成“指令遵循者”。没有 SFT，模型无法理解“请帮我总结以下内容”这样的指令。

### 4.1.3 为什么 SFT 是必需的

**（1）预训练模型的局限**

基础模型（Base Model）的训练目标是“预测下一个 token”，它只知道如何续写文本，不知道如何遵循指令。例如：

- 输入：“请翻译以下句子：Hello World”
- 基础模型可能输出：“你好世界”之后继续编造无关内容（如“这句话出自...”），而不是干净地只输出翻译结果。

**（2）SFT 的桥梁作用**

SFT 通过在大量“指令-回答”对上训练，让模型学会：

- 识别指令的意图
- 按照期望的格式输出
- 在合适的位置停止生成

**（3）SFT 与后续对齐的关系**

SFT 是 RLHF/DPO/GRPO 的前提。没有 SFT 的模型输出质量太差，无法进行偏好对比。SFT 提供了基础的指令遵循能力，偏好优化在此基础上进一步提升。

### 4.1.4 SFT 的核心目标

| 目标     | 说明                             |
| -------- | -------------------------------- |
| 指令理解 | 让模型学会理解并遵循自然语言指令 |
| 格式适配 | 让模型的输出符合特定格式要求     |
| 风格统一 | 让模型的回答风格符合业务需求     |
| 领域适应 | 让模型掌握特定领域的术语和知识   |
| 安全对齐 | 让模型拒绝有害请求，输出安全内容 |

## 4.2 SFT 的损失函数推导

### 4.2.1 自回归语言建模

SFT 的核心是自回归语言建模。给定序列 $y = (y_1, y_2, \dots, y_T)$，模型的目标是最大化整个序列的联合概率：

$$
P(y \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x)
$$

其中 $y_{<t} = (y_1, \dots, y_{t-1})$ 是之前生成的 token。

### 4.2.2 从最大似然到交叉熵损失

**（1）最大似然估计（MLE）**

训练目标是最大化训练数据上的似然：

$$
\mathcal{L}_{\text{MLE}} = \prod_{i=1}^{N} P_\theta(y_i \mid x_i)
$$

**（2）取对数**

为了数值稳定，取对数：

$$
\log \mathcal{L}_{\text{MLE}} = \sum_{i=1}^{N} \log P_\theta(y_i \mid x_i)
$$

代入自回归分解：

$$
\log \mathcal{L}_{\text{MLE}} = \sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid y_{i,<t}, x_i)
$$

**（3）转化为最小化负对数似然**

优化目标是最大化对数似然，等价于最小化负对数似然：

$$
\mathcal{L}_{\text{NLL}} = -\sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid y_{i,<t}, x_i)
$$

取平均：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid y_{i,<t}, x_i)
$$

**（4）与交叉熵的关系**

对于单个 token $y_t$，模型的输出是一个概率分布 $P_\theta(\cdot \mid y_{<t}, x) \in \mathbb{R}^V$，其中 $V$ 是词汇表大小。真实标签 $y_t$ 的 one-hot 表示为 $\mathbf{q}$，则单 token 的交叉熵为：

$$
H(\mathbf{q}, P) = -\sum_{v=1}^{V} q_v \log P_\theta(v \mid y_{<t}, x) = -\log P_\theta(y_t \mid y_{<t}, x)
$$

因此，SFT 损失就是所有 token 的交叉熵之和的平均。

### 4.2.3 损失函数的实现细节

**（1）如何屏蔽 input 部分的损失**

在 SFT 中，通常只对输出 $y$ 部分计算损失，不对输入 $x$ 部分计算。这是因为我们希望模型学习生成 $y$，而不是学习“生成” $x$。

实现上，通常将输入和输出拼接为一个序列：

$$
\text{seq} = [x_1, \dots, x_m, y_1, \dots, y_T]
$$

然后构造标签张量，将输入部分和 padding 部分的标签设为 `-100`（PyTorch 中 `CrossEntropyLoss` 的 `ignore_index`）：

```python
# 伪代码
labels = input_ids.clone()
labels[:, :m] = -100   # 屏蔽输入部分
labels[labels == pad_token_id] = -100   # 屏蔽padding
loss = CrossEntropyLoss(ignore_index=-100)(logits, labels)
```

**（2）损失函数的数学表达（含屏蔽）**

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_{i=1}^{N} |y_i|} \sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid \text{seq}_{i,<m+t})
$$

其中 $\text{seq}_{i,<m+t}$ 包含了输入 $x_i$ 和输出的前 $t-1$ 个 token。

**（3）权重加权**

某些场景下，不同的 token 或不同的样本可能有不同的重要性。可以引入权重 $w_{i,t}$：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_{i,t} w_{i,t}} \sum_{i=1}^{N} \sum_{t=1}^{|y_i|} w_{i,t} \log P_\theta(y_{i,t} \mid y_{i,<t}, x_i)
$$

例如，可以使用“token 级平均”而非“样本级平均”，避免长样本主导损失。

### 4.2.4 梯度推导

对于单个 token 的损失：

$$
\ell_t = -\log P_\theta(y_t \mid y_{<t}, x)
$$

设模型在时间步 $t$ 的 logits 为 $\mathbf{z} \in \mathbb{R}^V$，则：

$$
P_\theta(y_t \mid y_{<t}, x) = \frac{\exp(z_{y_t})}{\sum_{v=1}^{V} \exp(z_v)}
$$

对 logits 求梯度：

$$
\frac{\partial \ell_t}{\partial z_j} = P_\theta(j \mid y_{<t}, x) - \mathbb{1}[j = y_t]
$$

即：

$$
\nabla_{\mathbf{z}} \ell_t = \mathbf{p} - \mathbf{q}
$$

其中 $\mathbf{p}$ 是预测概率分布，$\mathbf{q}$ 是真实标签的 one-hot 向量。

这个梯度形式非常直观：如果模型预测正确（$p_{y_t} \to 1$），梯度趋近于 0；如果模型预测错误（$p_{y_t} \to 0$），梯度较大，推动模型修正。

## 4.3 SFT 的数据准备

### 4.3.1 数据格式

SFT 数据通常为（指令，输入，输出）三元组或（输入，输出）配对。以下是两种主流的数据格式：

**Alpaca 格式**（单轮/多轮指令微调）：

```json
[
  {
    "instruction": "计算这些物品的总费用。",
    "input": "输入：汽车 - $3000，衣服 - $100，书 - $20。",
    "output": "汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120。"
  }
]
```

**Alpaca 多轮对话格式**：

```json
[
  {
    "instruction": "今天的天气怎么样？",
    "input": "",
    "output": "今天的天气不错，是晴天。",
    "history": [
      ["今天会下雨吗？", "今天不会下雨，是个好天气。"],
      ["今天适合出去玩吗？", "非常适合，空气质量很好。"]
    ]
  }
]
```

**ShareGPT 格式**（支持更多角色种类）：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好，请介绍一下自己。"},
      {"from": "gpt", "value": "你好！我是一个AI助手..."},
      {"from": "human", "value": "你能做什么？"},
      {"from": "gpt", "value": "我可以帮你..."}
    ]
  }
]
```

### 4.3.2 数据质量要求

**准确性**：输出必须是正确的、经过审核的。错误的标注会直接损害模型性能。

**多样性**：覆盖不同的问题类型和难度。多样性在 SFT 阶段比数据质量更重要——在单一来源的小规模数据集上，基于数据质量的筛选方法比基于多样性的方法更有效，但在大规模数据上，多样性更为关键。

**一致性**：输出风格和格式应统一。如果同一类问题的回答风格不一致，模型会学到混乱的输出模式。

**充分性**：每类任务有足够的样本量。数据量决定了微调方法的上限。

### 4.3.3 数据规模建议

| 任务类型     | 建议样本量   | 说明                   |
| ------------ | ------------ | ---------------------- |
| 简单格式适配 | 500-2000     | 输出格式调整、风格统一 |
| 领域知识问答 | 5000-20000   | 需要覆盖领域核心知识点 |
| 复杂推理任务 | 20000-100000 | 需要大量多样化推理样本 |
| 通用指令微调 | 50000-500000 | 覆盖广泛的任务类型     |

**数据规模与超参数的关系**：

- **小数据集（<500 条）** ：训练轮次 2-3 轮、学习率 1e-5、批次大小 8、权重衰减 0.01、dropout 0.2。
- **中等数据集（500-2000 条）** ：训练轮次 3-5 轮、学习率 3e-5、批次大小 16。
- **大数据集（>2000 条）** ：训练轮次 1-3 轮、学习率 1e-5 至 5e-5、批次大小 32-128。

### 4.3.4 数据清洗与格式化

**数据清洗**：

- 去除重复样本
- 过滤低质量样本（如过短、过长、包含乱码）
- 统一标点和格式
- 处理敏感信息（PII 脱敏）

**数据格式化**：

根据使用的训练框架选择合适的数据格式。以 LLaMA-Factory 为例，需要在 `dataset_info.json` 中注册数据集：

```json
{
  "my_dataset": {
    "file_name": "train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }
}
```

### 4.3.5 数据划分

将数据划分为训练集、验证集和测试集。常用比例为：

- 训练集：80%-90%
- 验证集：5%-10%
- 测试集：5%-10%

**注意**：验证集和测试集必须与训练集在分布上一致，且不能有重叠。

### 4.3.6 数据构建策略

**（1）人工标注**

- 适合高质量、小规模数据
- 成本高，每条约 1-10 元
- 质量最有保障

**（2）模型合成（Distillation）**

使用更强的模型（如 GPT-4）生成回答，再人工审核：

```python
import openai

def generate_sft_data(instruction, input_text=""):
    prompt = f"""请根据以下指令生成高质量回答：

指令：{instruction}
输入：{input_text}

要求：
1. 回答准确、完整
2. 格式规范
3. 语言自然流畅

回答："""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content
```

**（3）Self-Instruct**

让模型自己生成指令和回答，再过滤低质量样本：

```
步骤1：用少量种子指令，让模型生成新指令
步骤2：让模型为新指令生成回答
步骤3：用规则和模型过滤低质量样本
步骤4：将高质量样本加入训练集
```

**（4）数据增强**

- 同义改写
- 回译（Back Translation）
- 格式变换

## 4.4 SFT 的训练配置

### 4.4.1 常用超参数

基于 2026 年的最佳实践，SFT 的推荐超参数如下：

| 参数         | 推荐值             | 说明                    |
| ------------ | ------------------ | ----------------------- |
| 学习率       | 1e-5 至 5e-5       | 比预训练小 1-2 个数量级 |
| 有效批次大小 | 16-128             | 根据显存调整            |
| 训练轮数     | 1-5                | 过多容易过拟合          |
| 学习率调度   | 余弦退火（cosine） | 带 warmup               |
| Warmup 比例  | 0.1                | 前 10% 步数线性预热     |
| 权重衰减     | 0.01-0.1           | 防止过拟合              |
| Dropout      | 0.05-0.2           | 小数据集取较大值        |
| 优化器       | AdamW              | 标准选择                |
| 精度         | bf16               | 比 fp16 更稳定          |

**学习率的选择逻辑**：

- 学习率过大 → 模型可能遗忘预训练知识，输出不稳定
- 学习率过小 → 训练缓慢，可能无法充分适应任务
- 推荐从 1e-5 开始尝试，如果效果不佳再调整

**训练轮数的选择**：

- 1 轮通常足够让模型学会输出格式
- 2-3 轮适合需要学习新知识的场景
- 超过 5 轮容易过拟合，尤其是小数据集

### 4.4.2 学习率调度推导

**（1）余弦退火 + Warmup**

训练时使用余弦退火学习率，公式为：

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}} \pi \right) \right)
$$

其中：

- $\eta_{\max}$：最大学习率
- $\eta_{\min}$：最小学习率（通常为 0 或接近 0）
- $t$：当前步数
- $t_{\text{warmup}}$：warmup 步数
- $T$：总步数

**（2）Warmup 阶段**

在前 $t_{\text{warmup}}$ 步，学习率线性增加：

$$
\eta_t = \eta_{\max} \cdot \frac{t}{t_{\text{warmup}}}, \quad t \leq t_{\text{warmup}}
$$

**（3）为什么需要 Warmup**

- 训练初期，梯度可能很大，大学习率会导致训练不稳定
- Warmup 让模型逐渐适应训练数据，避免早期震荡
- 对于预训练模型尤其重要，因为预训练模型的参数已经在一个复杂的损失面上

### 4.4.3 梯度累积的数学原理

当显存不足以支持大批次时，使用梯度累积模拟大批次。设：

- 小批次大小：$b$
- 梯度累积步数：$k$
- 有效批次大小：$B = b \times k$

标准的 SGD 更新（大批次）为：

$$
\theta \leftarrow \theta - \eta \cdot \frac{1}{B} \sum_{i=1}^{B} \nabla \ell_i
$$

梯度累积的更新（小批次 + 累积）为：

$$
\theta \leftarrow \theta - \eta \cdot \frac{1}{k} \sum_{j=1}^{k} \frac{1}{b} \sum_{i=1}^{b} \nabla \ell_{j,i}
$$

可以证明两者等价：

$$
\frac{1}{k} \sum_{j=1}^{k} \frac{1}{b} \sum_{i=1}^{b} \nabla \ell_{j,i} = \frac{1}{B} \sum_{i=1}^{B} \nabla \ell_i
$$

因此，梯度累积在数学上等价于大批次训练，只是显存占用更少（每次只处理 $b$ 个样本）。

### 4.4.4 灾难性遗忘的缓解

SFT 可能导致模型遗忘预训练阶段学到的通用能力。最新的缓解策略包括：

**（1）使用 PEFT 方法**

LoRA 及其变体通过冻结大部分参数，有效缓解了灾难性遗忘。L2-LoRA 进一步引入层特定的 L2 正则化，在微调过程中约束 LoRA 权重的变化幅度，保留更多预训练知识。

**（2）混合通用数据**

在训练数据中混合一定比例的通用数据，让模型在适应新任务的同时保持通用能力。建议的混合比例为 60%-80% 领域数据 + 20%-40% 通用数据。

**（3）使用较小的学习率和较少的训练轮数**

较小的学习率减少了参数更新的幅度，较少的训练轮数避免了过度适应。

**（4）正则化技术**

- 权重衰减（Weight Decay）
- 层特定 L2 正则化（L2-LoRA）
- 梯度裁剪

### 4.4.5 训练监控

**监控指标**：

- 训练损失（Training Loss）
- 验证损失（Validation Loss）
- 学习率变化曲线
- 梯度范数

**监控工具**：

- Weights & Biases（推荐）
- TensorBoard
- MLflow

**早停策略**：

当验证损失连续 N 个 epoch 不再下降时，停止训练。通常 N 取 2-3。

### 4.4.6 训练效率优化

**（1）梯度累积**

当显存不足以支持大批次时，使用梯度累积模拟大批次：

```python
# 有效批次大小 = batch_size * gradient_accumulation_steps
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 有效批次 = 32
    ...
)
```

**（2）序列打包（Sequence Packing）**

将多个短序列拼接成一个长序列，减少 padding 浪费，提升训练效率。LLaMA-Factory 等框架支持此功能。

数学上，将多个短序列 $(s_1, s_2, \dots, s_k)$ 拼接为：

$$
\text{packed} = [s_1; s_2; \dots; s_k]
$$

其中每个 $s_i$ 长度不同。序列打包将多个样本拼接到一起，总长度接近 `max_length`，大幅减少 padding 开销。

**（3）混合精度训练**

使用 bf16 或 fp16 混合精度，减少显存占用并加速训练。bf16 比 fp16 更稳定，推荐优先使用。

**（4）FlashAttention**

使用 FlashAttention 优化注意力计算，显著减少显存占用并加速训练。

## 4.5 SFT 的代码实现

### 4.5.1 使用 Hugging Face TRL 的 SFTTrainer

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# 1. 加载模型和分词器
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bf16")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载数据集
dataset = load_dataset("json", data_files="train.json", split="train")

# 3. 配置训练参数
training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
)

# 4. 创建训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 5. 开始训练
trainer.train()

# 6. 保存模型
trainer.save_model("./sft_final")
```

### 4.5.2 使用 LLaMA-Factory 进行 SFT

LLaMA-Factory 提供了更简洁的配置方式：

**命令行训练**：

```bash
llamafactory-cli train \
    --stage sft \
    --model_name_or_path meta-llama/Llama-3-8B \
    --dataset my_dataset \
    --template llama3 \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --output_dir ./sft_lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --logging_steps 10 \
    --save_steps 500
```

**Web UI 训练**：

```bash
llamafactory-cli webui
```

然后在浏览器中配置数据集、模型和超参数，点击“开始训练”。

### 4.5.3 使用 Unsloth 加速训练

Unsloth 提供比标准 Hugging Face 实现快 2-5 倍的微调速度：

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 加载模型（自动优化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3-8B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

# 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        bf16=True,
        output_dir="./unsloth_output",
    ),
)

trainer.train()
```

## 4.6 SFT 的评估

### 4.6.1 自动评估指标

| 任务类型 | 常用指标               | 说明                |
| -------- | ---------------------- | ------------------- |
| 分类     | 准确率、F1、AUC        | 适用于分类任务      |
| 生成     | BLEU、ROUGE、BERTScore | 适用于生成任务      |
| 问答     | EM（精确匹配）、F1     | 适用于抽取式问答    |
| 推理     | 准确率、通过率         | 适用于数学/代码推理 |

**（1）BLEU 分数**

BLEU 衡量生成文本与参考文本的 n-gram 重叠度：

$$
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)
$$

其中 $p_n$ 是 n-gram 精确率，$w_n$ 是权重，BP 是长度惩罚：

$$
\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \leq r \end{cases}
$$

其中 $c$ 是生成文本长度，$r$ 是参考文本长度。

**（2）ROUGE 分数**

ROUGE 衡量召回率：

$$
\text{ROUGE-N} = \frac{\sum_{S \in \text{Ref}} \sum_{\text{n-gram} \in S} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{S \in \text{Ref}} \sum_{\text{n-gram} \in S} \text{Count}(\text{n-gram})}
$$

**（3）BERTScore**

BERTScore 使用 BERT 嵌入计算生成文本和参考文本的语义相似度：

$$
\text{BERTScore} = \frac{1}{|y|} \sum_{y_i \in y} \max_{\hat{y}_j \in \hat{y}} \text{cos}(\mathbf{e}_{y_i}, \mathbf{e}_{\hat{y}_j})
$$

其中 $\mathbf{e}$ 是 BERT 嵌入。

### 4.6.2 LLM-as-Judge

使用更强的 LLM（如 GPT-4、Claude）作为评判者，对模型输出进行质量评估。适用于开放式生成任务的评估。

```python
import openai

def llm_judge(question, answer, reference=None):
    prompt = f"""
    请评估以下回答的质量，从 1-5 分打分（5 分最好）。
    
    问题：{question}
    回答：{answer}
    {'参考答案：' + reference if reference else ''}
    
    请输出分数和理由。
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### 4.6.3 人工评估

对于关键应用，人工评估仍然是金标准。人工评估的维度包括：

- **准确性**：回答是否正确
- **相关性**：回答是否与问题相关
- **流畅性**：语言是否自然流畅
- **安全性**：是否包含有害内容
- **风格一致性**：是否符合期望的风格

### 4.6.4 评估流程

```
1. 在测试集上运行模型，收集输出
2. 计算自动指标
3. 使用 LLM-as-Judge 进行批量评估
4. 人工抽查关键样本
5. 分析错误模式，识别薄弱环节
6. 根据评估结果迭代优化
```

## 4.7 SFT 实战注意事项

### 4.7.1 数据层面的注意事项

- **数据质量优先于数量**：1000 条高质量数据可能优于 10000 条低质量数据。
- **覆盖边界情况**：确保训练数据包含异常输入和边界情况。
- **避免数据泄露**：测试集数据不能出现在训练集中。
- **保持格式一致**：所有样本的输出格式应统一。

### 4.7.2 训练层面的注意事项

- **从小规模实验开始**：先用 100-200 条数据验证流程，再扩展到全量数据。
- **监控验证损失**：验证损失上升时立即停止训练。
- **保存检查点**：定期保存模型，便于回滚和选择最佳模型。
- **记录实验配置**：使用 W&B 或 TensorBoard 记录所有超参数和结果。

### 4.7.3 部署层面的注意事项

- **LoRA 权重合并**：推理前将 LoRA 权重合并回原模型，消除推理延迟。
- **量化部署**：将模型量化为 4bit 或 8bit，降低推理显存需求。
- **推理框架选择**：使用 vLLM、TGI 等推理框架，提升吞吐量。
- **A/B 测试**：在生产环境中对比微调前后的模型表现。

## 4.8 SFT 的常见问题与解决方案

### 4.8.1 模型过拟合

**表现**：训练损失持续下降，但验证损失开始上升；模型在训练集上输出准确，在新样本上输出不稳定。

**解决方案**：

- 减少训练轮数
- 增加 Dropout
- 增大权重衰减
- 使用数据增强
- 使用早停

### 4.8.2 模型不遵循指令

**表现**：模型忽略用户指令，继续按预训练风格续写文本。

**解决方案**：

- 检查数据格式是否正确（指令和输出是否明确分隔）
- 增加指令多样性
- 使用更大的学习率
- 增加训练轮数

### 4.8.3 输出重复

**表现**：模型生成重复的内容，如“这是一个好问题，这是一个好问题，这是一个好问题...”

**解决方案**：

- 检查训练数据是否包含大量重复内容
- 使用 repetition penalty 推理时惩罚重复
- 增加训练数据多样性

### 4.8.4 灾难性遗忘

**表现**：微调后模型在目标任务上表现良好，但通用能力下降。

**解决方案**：

- 使用 PEFT（LoRA）
- 混合通用数据
- 使用更小的学习率
- 使用 L2-LoRA 正则化

### 4.8.5 训练不稳定

**表现**：损失波动大，甚至出现 NaN。

**解决方案**：

- 使用 bf16 而非 fp16
- 添加梯度裁剪（`max_grad_norm=1.0`）
- 减少学习率
- 使用更长的 warmup
- 检查数据中是否有异常长或异常短的样本



# 五、偏好优化与对齐

## 5.1 RLHF（Reinforcement Learning from Human Feedback）

### 5.1.1 核心流程

RLHF（基于人类反馈的强化学习）由 Christiano 等人于 2017 年提出，并由 InstructGPT（2022）发扬光大，是 ChatGPT 成功的关键技术之一。其核心思想是：先让模型学会“什么是好的回答”，再用强化学习让模型生成更多这样的回答。

RLHF 通常包含三个步骤：

**步骤一：监督微调（SFT）**

使用人类示范数据对预训练模型进行微调，得到初始策略模型 $\pi_{\text{SFT}}$。这一步让模型具备基本的指令遵循能力。

**步骤二：训练奖励模型（Reward Model, RM）**

收集人类对模型输出的偏好比较数据。具体做法是：对于同一个提示 $x$，让 SFT 模型生成多个回答 $(y_1, y_2, \dots, y_K)$，然后让人类标注者对这些回答进行排序。通常简化为两两比较：给定 $(x, y_w, y_l)$，其中 $y_w$ 是人类更喜欢的回答（winner），$y_l$ 是较差的回答（loser）。

奖励模型 $r_\phi(x, y)$ 的目标是预测人类偏好。训练损失通常采用 Bradley-Terry 模型：

$$
P(y_w \succ y_l \mid x) = \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right)
$$

其中 $\sigma$ 是 Sigmoid 函数。奖励模型的损失函数为：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
$$

**步骤三：PPO 强化学习**

以 SFT 模型为初始策略 $\pi_\theta$，以奖励模型 $r_\phi$ 为环境奖励，使用 PPO（Proximal Policy Optimization）算法优化策略，同时加入 KL 惩罚防止策略偏离 SFT 模型太远。

### 5.1.2 奖励模型训练：Bradley-Terry 模型与损失推导

**（1）Bradley-Terry 模型**

Bradley-Terry 模型最初用于描述成对比较的概率。假设每个回答 $y$ 有一个潜在的“质量分数” $r^*(x, y)$，则人类偏好 $y_w$ 胜过 $y_l$ 的概率为：

$$
P(y_w \succ y_l \mid x) = \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))}
$$

将分子分母同除以 $\exp(r^*(x, y_w))$，得到：

$$
P(y_w \succ y_l \mid x) = \frac{1}{1 + \exp(-(r^*(x, y_w) - r^*(x, y_l)))} = \sigma\left( r^*(x, y_w) - r^*(x, y_l) \right)
$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 Sigmoid 函数。

**（2）奖励模型的损失函数**

我们用参数化的奖励模型 $r_\phi(x, y)$ 去拟合真实质量分数 $r^*(x, y)$。对于偏好数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$，最大化偏好数据的似然：

$$
\mathcal{L}_{\text{likelihood}}(\phi) = \prod_{i=1}^N P(y_w^{(i)} \succ y_l^{(i)} \mid x^{(i)})
$$

取对数：

$$
\log \mathcal{L}_{\text{likelihood}}(\phi) = \sum_{i=1}^N \log \sigma\left( r_\phi(x^{(i)}, y_w^{(i)}) - r_\phi(x^{(i)}, y_l^{(i)}) \right)
$$

最大化对数似然等价于最小化负对数似然：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\sum_{i=1}^N \log \sigma\left( r_\phi(x^{(i)}, y_w^{(i)}) - r_\phi(x^{(i)}, y_l^{(i)}) \right)
$$

取期望形式：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
$$

**（3）梯度分析**

对 $\mathcal{L}_{\text{RM}}$ 关于 $r_\phi(x, y_w)$ 求梯度：

$$
\frac{\partial \mathcal{L}_{\text{RM}}}{\partial r_\phi(x, y_w)} = -\frac{\sigma'(z)}{\sigma(z)} = -(1 - \sigma(z)) = \sigma(z) - 1
$$

其中 $z = r_\phi(x, y_w) - r_\phi(x, y_l)$。

对 $r_\phi(x, y_l)$ 求梯度：

$$
\frac{\partial \mathcal{L}_{\text{RM}}}{\partial r_\phi(x, y_l)} = \frac{\sigma'(z)}{\sigma(z)} = 1 - \sigma(z)
$$

直观理解：如果模型预测偏好正确（即 $\sigma(z) \to 1$），梯度趋近于 0；如果预测错误（即 $\sigma(z) \to 0$），梯度会推动 $r_\phi(x, y_w)$ 增大，$r_\phi(x, y_l)$ 减小。

### 5.1.3 策略优化目标：KL 约束的推导

在得到奖励模型后，我们希望优化策略 $\pi_\theta$ 使得期望奖励最大，同时不要偏离参考策略 $\pi_{\text{ref}}$ 太远。优化目标为：

$$
\max_{\pi_\theta} \ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)} \left[ r_\phi(x, y) \right] - \beta \cdot \text{KL}\left( \pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x) \right)
$$

其中 KL 散度定义为：

$$
\text{KL}\left( \pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x) \right) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

因此优化目标可以写为：

$$
\max_{\pi_\theta} \ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

**为什么需要 KL 惩罚？**

- 奖励模型只是在有限偏好数据上训练的近似，存在误差。
- 如果策略过度优化奖励模型，会利用奖励模型的弱点（奖励黑客），生成高奖励但无意义的输出。
- KL 惩罚约束策略不要偏离参考模型太远，保持生成质量。

**β 的作用**：

- $\beta$ 越大，策略越接近参考模型，越保守。
- $\beta$ 越小，策略越激进，越可能利用奖励模型的漏洞。

### 5.1.4 PPO 算法详解与损失推导

PPO（Proximal Policy Optimization）是一种策略梯度算法，核心思想是限制每次策略更新的幅度，避免训练不稳定。

**（1）策略梯度基础**

策略梯度的目标是最大化期望奖励：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$

其梯度为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]
$$

其中 $\hat{A}_t$ 是优势函数，衡量动作 $a_t$ 相对于平均水平的优劣。

**（2）重要性采样**

为了复用旧策略采集的数据，PPO 使用重要性采样：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} \hat{A}_t \right]
$$

令 $\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$，则目标函数为：

$$
J^{\text{IS}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{old}}} \left[ \rho_t(\theta) \hat{A}_t \right]
$$

**（3）PPO 的裁剪目标**

重要性采样比率 $\rho_t(\theta)$ 可能过大，导致更新步长过大。PPO 引入裁剪：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_{t} \left[ \min\left( \rho_t(\theta) \hat{A}_t, \ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $\epsilon$ 是裁剪参数，通常取 0.1 或 0.2。$\text{clip}(x, a, b)$ 将 $x$ 限制在 $[a, b]$ 范围内。

**裁剪的直观解释**：

- 当 $\hat{A}_t > 0$（动作好）时，我们希望增大 $\pi_\theta(a_t \mid s_t)$，但 $\rho_t$ 最多增大到 $1+\epsilon$。
- 当 $\hat{A}_t < 0$（动作差）时，我们希望减小 $\pi_\theta(a_t \mid s_t)$，但 $\rho_t$ 最多减小到 $1-\epsilon$。
- 超出裁剪范围的部分不再产生梯度，避免过度更新。

**（4）优势函数估计（GAE）**

优势函数 $\hat{A}_t$ 使用 GAE（Generalized Advantage Estimation）计算：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$ 是 TD 误差，$\gamma$ 是折扣因子，$\lambda$ 是 GAE 参数。$V_\psi(s_t)$ 是 Critic 网络估计的状态价值。

在 RLHF 中，状态 $s_t$ 是提示 $x$ 加上已生成的 token 序列，动作 $a_t$ 是下一个 token，奖励 $r_t$ 通常只在序列末尾给出（即 $r_T = r_\phi(x, y)$，其他步为 0）。

**（5）完整的 PPO 损失**

PPO 的完整损失包括三部分：

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \mathcal{L}^{\text{CLIP}}_t(\theta) - c_1 \mathcal{L}^{\text{VF}}_t(\psi) + c_2 \mathcal{S}[\pi_\theta](s_t) \right]
$$

其中：

- $\mathcal{L}^{\text{CLIP}}_t(\theta)$：裁剪的策略损失
- $\mathcal{L}^{\text{VF}}_t(\psi) = (V_\psi(s_t) - V_t^{\text{target}})^2$：价值函数的均方误差
- $\mathcal{S}[\pi_\theta](s_t)$：策略的熵，用于鼓励探索
- $c_1, c_2$：权重系数

在 RLHF 中，通常还会加入 KL 惩罚：

$$
r_t^{\text{total}} = r_t - \beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}
$$

### 5.1.5 实操挑战

RLHF 虽然强大，但在实践中面临诸多挑战：

**（1）显存需求大**

PPO 需要同时加载四个模型：

- 策略模型 $\pi_\theta$（可训练）
- 参考模型 $\pi_{\text{ref}}$（冻结）
- 奖励模型 $r_\phi$（冻结）
- Critic 网络 $V_\psi$（可训练）

以 70B 模型为例，仅模型权重就需要约 280GB 显存（FP16），加上优化器状态和激活值，总需求超过 1TB。

**（2）奖励黑客（Reward Hacking）**

策略会利用奖励模型的弱点，生成高奖励但无意义的输出。例如：

- 生成冗长的回答（奖励模型可能偏好长回答）
- 过度使用项目符号和 Markdown 标题
- 重复训练数据中的内容

**（3）分布漂移**

奖励模型在 SFT 模型的输出分布上训练，当策略更新后，生成的分布偏离训练分布，奖励模型的可靠性下降。

**（4）超参数脆弱**

PPO 对超参数非常敏感，裁剪比 $\epsilon$、KL 系数 $\beta$、价值损失权重、学习率等需要精细调节。不同任务的最佳超参数差异很大。

**（5）训练不稳定**

策略更新幅度过大会导致崩溃，需要精心设计 KL 约束和梯度裁剪。

## 5.2 DPO（Direct Preference Optimization）

### 5.2.1 核心思想

DPO（直接偏好优化）由 Rafailov 等人于 2023 年提出，是一种直接对齐算法（Direct Alignment Algorithm, DAA）。它的核心洞察是：**RLHF 的约束优化目标存在闭式最优解，且该最优解可以转化为一个简单的分类损失，从而跳过奖励模型和强化学习优化器，直接在偏好数据上优化语言模型。**

DPO 的出发点是 RLHF 的 KL 约束优化目标：

$$
\max_{\pi_\theta} \ \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

其中 $r(x, y)$ 是真实奖励函数（在 RLHF 中由奖励模型近似）。

### 5.2.2 从 RLHF 到 DPO 的完整数学推导

#### 5.2.2.1 最优策略的闭式解推导

考虑优化问题：

$$
\max_{\pi} \ \mathbb{E}_{y \sim \pi(y \mid x)} \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

约束条件为 $\sum_y \pi(y \mid x) = 1$，$\pi(y \mid x) \geq 0$。

使用拉格朗日乘子法，构造拉格朗日函数：

$$
\mathcal{L}(\pi, \lambda) = \sum_y \pi(y \mid x) \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] + \lambda \left( 1 - \sum_y \pi(y \mid x) \right)
$$

对 $\pi(y \mid x)$ 求偏导并令其为 0：

$$
\frac{\partial \mathcal{L}}{\partial \pi(y \mid x)} = r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \beta - \lambda = 0
$$

整理得：

$$
\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = r(x, y) - \beta - \lambda
$$

$$
\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \frac{1}{\beta} r(x, y) - 1 - \frac{\lambda}{\beta}
$$

$$
\pi(y \mid x) = \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) - 1 - \frac{\lambda}{\beta} \right)
$$

令 $Z(x) = \exp\left( 1 + \frac{\lambda}{\beta} \right) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$，则最优策略为：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

其中 $Z(x)$ 是归一化常数（配分函数）。

#### 5.2.2.2 奖励函数的反解

对最优策略的表达式取对数：

$$
\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)
$$

整理得：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

#### 5.2.2.3 代入 Bradley-Terry 得到 DPO 损失

根据 Bradley-Terry 模型，人类偏好 $y_w \succ y_l$ 的概率为：

$$
P(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)
$$

将 $r(x, y)$ 的表达式代入：

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

注意 $\beta \log Z(x)$ 项在相减时抵消。

因此：

$$
P(y_w \succ y_l \mid x) = \sigma\left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
$$

我们用参数化的策略 $\pi_\theta$ 替代 $\pi^*$，最大化偏好数据的似然，等价于最小化负对数似然：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

这就是 DPO 的最终损失函数。

#### 5.2.2.4 梯度分析

对 DPO 损失求梯度：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}_{(x, y_w, y_l)} \left[ \sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right) \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right) \right]
$$

其中 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ 是隐式奖励。

梯度直观含义：

- 当模型错误地偏好 $y_l$ 时（即 $\hat{r}_\theta(x, y_l) > \hat{r}_\theta(x, y_w)$），$\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 较大，梯度会增大 $y_w$ 的概率，降低 $y_l$ 的概率。
- 当模型已经正确偏好 $y_w$ 时（即 $\hat{r}_\theta(x, y_w) > \hat{r}_\theta(x, y_l)$），$\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 趋近于 0，梯度趋近于 0。

### 5.2.3 训练数据格式

DPO 使用三元组数据：`(prompt, chosen, rejected)`，即给定提示词、好的回答和差的回答。

```json
[
  {
    "prompt": "什么是机器学习？",
    "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进，而无需显式编程。",
    "rejected": "机器学习就是让电脑自己学东西。"
  }
]
```

### 5.2.4 与 RLHF 的对比

| 维度               | RLHF (PPO)    | DPO           |
| ------------------ | ------------- | ------------- |
| 需要奖励模型       | 是            | 否            |
| 需要强化学习优化器 | 是            | 否            |
| 需要 Critic 网络   | 是            | 否            |
| 训练稳定性         | 超参数敏感    | 更稳定        |
| 显存需求           | 高（4个模型） | 低（2个模型） |
| 实现复杂度         | 高            | 低            |
| 性能               | 优秀          | 相当或更好    |
| 训练速度           | 慢            | 快            |

### 5.2.5 代码实现

使用 Hugging Face TRL 的 DPOTrainer：

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# 1. 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# 2. 加载偏好数据集
dataset = load_dataset("json", data_files="dpo_data.json", split="train")

# 3. 配置训练参数
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    beta=0.1,  # KL 惩罚系数
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 4. 创建训练器
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 5. 开始训练
trainer.train()
```

## 5.3 GRPO（Group Relative Policy Optimization）

### 5.3.1 核心原理

GRPO（组相对策略优化）由 DeepSeek 于 2024 年提出（DeepSeekMath 论文），是 PPO 的简化变体。其核心创新是：**移除了 PPO 中的 Critic 网络（价值函数），使用一组采样输出的平均奖励作为基线来估计优势函数。** 这大幅降低了显存需求和计算复杂度。

在 PPO 中，优势函数 $\hat{A}_t$ 需要 Critic 网络 $V_\psi(s_t)$ 来估计基线。GRPO 则通过对同一提示采样多个输出，用组内奖励的统计量作为基线。

### 5.3.2 优势估计推导

对于每个查询 $q$，GRPO 采样一组输出 $\{o_1, o_2, \dots, o_G\}$，计算每个输出的奖励 $r_i = r(q, o_i)$。

**（1）基线估计**

在标准策略梯度中，优势函数 $A(s, a) = Q(s, a) - V(s)$，其中 $V(s)$ 是状态价值，作为基线。GRPO 用组内平均奖励作为基线的估计：

$$
V(q) \approx \text{mean}(\{r_1, r_2, \dots, r_G\}) = \frac{1}{G} \sum_{i=1}^G r_i
$$

**（2）优势计算**

因此，每个输出的优势为：

$$
A_i = r_i - \text{mean}(\{r_1, \dots, r_G\})
$$

为了消除不同任务间奖励尺度的影响，进行标准化：

$$
A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}
$$

其中 $\text{std}(\cdot)$ 是组内奖励的标准差。

**（3）直观理解**

- 如果一个输出的奖励高于组内平均，则其优势为正，策略会倾向于增加该输出的概率。
- 如果一个输出的奖励低于组内平均，则其优势为负，策略会倾向于降低该输出的概率。
- 标准差归一化使得优势在不同任务间尺度一致，便于统一超参数。

### 5.3.3 优化目标

GRPO 的优化目标与 PPO 类似，但使用组内优势：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i(\theta) A_i, \ \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}}) \right]
$$

其中 $\rho_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}$ 是重要性采样比率。

### 5.3.4 与 PPO 的对比

| 维度             | PPO          | GRPO                 |
| ---------------- | ------------ | -------------------- |
| 需要 Critic 网络 | 是           | 否                   |
| 优势估计方式     | GAE + Critic | 组内奖励统计         |
| 显存需求         | 高           | 低（省去 Critic）    |
| 训练稳定性       | 较敏感       | 更稳定               |
| 适用场景         | 通用 RLHF    | 推理任务、可验证奖励 |
| 代表应用         | ChatGPT      | DeepSeek-R1          |

### 5.3.5 代码实现

使用 Hugging Face TRL 的 GRPOTrainer：

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

def reward_function(completions, prompts, **kwargs):
    """自定义奖励函数，返回每个输出的奖励"""
    rewards = []
    for completion in completions:
        # 示例：根据答案是否正确计算奖励
        if "正确答案" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_generations=8,  # 组大小 G
    beta=0.04,          # KL 惩罚系数
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_function,
    tokenizer=tokenizer,
)

trainer.train()
```

## 5.4 RLAIF（Reinforcement Learning from AI Feedback）

### 5.4.1 核心原理

RLAIF（基于 AI 反馈的强化学习）使用 AI 模型（通常是更强的 LLM）代替人类进行偏好标注，从而大幅降低偏好数据的获取成本。其核心流程与 RLHF 相同，只是将人类标注者替换为 AI 标注者。

**具体做法**：

1. 使用强 LLM（如 GPT-4、Claude）对模型输出进行评分或比较。
2. 用 AI 标注的偏好数据训练奖励模型。
3. 使用 PPO 或 DPO 优化策略。

**优势**：

- 成本极低：AI 标注成本远低于人工标注
- 速度快：可以大规模生成偏好数据
- 一致性高：AI 标注标准统一

**挑战**：

- 可能引入 AI 自身的偏见
- 对强 LLM 的依赖
- 需要设计良好的提示词来引导 AI 标注

### 5.4.2 与 RLHF 对比

| 维度     | RLHF                       | RLAIF                      |
| -------- | -------------------------- | -------------------------- |
| 标注者   | 人类                       | AI 模型                    |
| 成本     | 高                         | 低                         |
| 速度     | 慢                         | 快                         |
| 一致性   | 较低（人类标注者间有差异） | 高（同一 AI 模型标准统一） |
| 偏见     | 人类偏见                   | AI 偏见                    |
| 适用场景 | 高质量对齐                 | 大规模对齐、成本敏感       |

### 5.4.3 代码示例

```python
import openai

def ai_preference_judge(prompt, response_a, response_b):
    """使用 GPT-4 判断哪个回答更好"""
    judge_prompt = f"""
    请判断以下两个回答哪个更好。

    问题：{prompt}
    回答 A：{response_a}
    回答 B：{response_b}

    请输出 "A" 或 "B"，并说明理由。
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

## 5.5 RLVR（Reinforcement Learning with Verifiable Rewards）

### 5.5.1 核心原理

RLVR（基于可验证奖励的强化学习）使用可程序化验证的奖励信号，无需人类标注或奖励模型。它特别适合数学、代码、逻辑推理等任务，其中答案的正确性可以被自动验证。

**典型奖励信号**：

- 数学题：最终答案是否与标准答案匹配
- 代码生成：代码是否通过单元测试
- 逻辑推理：结论是否有效
- 格式要求：输出是否符合指定格式

**RLVR 的优势**：

- 奖励信号客观、准确
- 无需训练奖励模型
- 可以大规模自动化
- 在推理任务上效果显著

**RLVR 与 GRPO 的结合**：

RLVR 通常与 GRPO 结合使用，因为 GRPO 不需要 Critic 网络，且组内奖励统计天然适合可验证奖励的场景。

### 5.5.2 奖励函数设计

以数学题为例，奖励函数可以设计为：

```python
def verifiable_reward(prompt, completion):
    """可验证奖励函数示例：数学题"""
    import re
    # 从 completion 中提取答案
    answer_match = re.search(r"答案[:：]\s*(\d+)", completion)
    if not answer_match:
        return 0.0
    predicted = int(answer_match.group(1))
    
    # 从 prompt 中提取标准答案
    ground_truth = extract_ground_truth(prompt)
    
    # 比较答案
    if predicted == ground_truth:
        return 1.0
    else:
        return 0.0

def extract_ground_truth(prompt):
    """从提示词中提取标准答案（实际中可能来自数据集）"""
    import re
    match = re.search(r"标准答案[:：]\s*(\d+)", prompt)
    return int(match.group(1)) if match else None
```

### 5.5.3 与 GRPO 结合

RLVR + GRPO 的训练流程：

1. 对每个数学题提示，采样 G 个输出。
2. 用可验证奖励函数计算每个输出的奖励（0 或 1）。
3. 计算组内优势：$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$。
4. 使用 GRPO 损失更新策略。

这种方法在 DeepSeek-R1 等推理模型上取得了显著效果。

## 5.6 对齐方法演进脉络

大模型对齐方法经历了清晰的演进路径：

```
PPO + RLHF（2022，ChatGPT）
    ↓ 去除奖励模型
DPO（2023，直接偏好优化）
    ↓ 去除 Critic 网络
GRPO（2024，组相对策略优化）
    ↓ 可验证奖励 + AI 反馈
RLVR / RLAIF（2025-2026，可验证奖励与 AI 反馈）
    ↓ 多智能体协同
多智能体 RL（2026+，MARL 框架）
```

**演进逻辑**：

1. **RLHF**：引入人类反馈，但需要奖励模型和 PPO，复杂且昂贵。
2. **DPO**：发现 RLHF 有闭式解，直接优化策略，去掉奖励模型和 RL。
3. **GRPO**：去掉 Critic 网络，用组内统计估计优势，大幅降低显存。
4. **RLVR/RLAIF**：用可验证奖励或 AI 反馈替代人类反馈，进一步降低成本。
5. **多智能体 RL**：将 RL 扩展到多智能体协作场景，处理更复杂的任务。

**各方法核心对比**：

| 方法  | 奖励来源           | 是否需要奖励模型 | 是否需要 Critic | 典型应用       |
| ----- | ------------------ | ---------------- | --------------- | -------------- |
| RLHF  | 人类偏好           | 是               | 是              | ChatGPT        |
| DPO   | 人类偏好           | 否               | 否              | 开源对齐       |
| GRPO  | 可验证奖励/AI 反馈 | 否               | 否              | DeepSeek-R1    |
| RLAIF | AI 反馈            | 可选             | 可选            | 大规模对齐     |
| RLVR  | 程序化验证         | 否               | 否              | 数学、代码推理 |





# 六、微调前沿技术与研究方向

## 6.1 CTR-LoRA（Curvature-Aware and Trust-Region Guided Low-Rank Adaptation）

### 6.1.1 核心思想

CTR-LoRA（曲率感知与信任域引导的低秩适应）针对 LoRA 在高秩设置下训练不稳定、低秩设置下表达能力不足的问题，引入两个关键机制：

1. **曲率感知**：利用损失函数的二阶信息（Hessian）指导 LoRA 参数的更新方向。
2. **信任域引导**：约束每次参数更新的幅度，确保训练稳定。

这两个机制共同作用，使 LoRA 在保持参数高效的同时，获得更强的表达能力和更稳定的训练过程。

### 6.1.2 曲率感知的数学推导

**（1）损失函数的二阶泰勒展开**

设当前参数为 $\theta$（包含 LoRA 的 $A$ 和 $B$），损失函数为 $\mathcal{L}(\theta)$。在 $\theta_t$ 处的二阶泰勒展开为：

$$
\mathcal{L}(\theta_t + \Delta\theta) \approx \mathcal{L}(\theta_t) + \nabla \mathcal{L}(\theta_t)^\top \Delta\theta + \frac{1}{2} \Delta\theta^\top H_t \Delta\theta
$$

其中 $H_t = \nabla^2 \mathcal{L}(\theta_t)$ 是 Hessian 矩阵。

**（2）最优更新方向**

最小化上述二阶近似，对 $\Delta\theta$ 求导并令为 0：

$$
\nabla \mathcal{L}(\theta_t) + H_t \Delta\theta = 0
$$

解得牛顿更新方向：

$$
\Delta\theta^* = -H_t^{-1} \nabla \mathcal{L}(\theta_t)
$$

牛顿法收敛快，但计算 $H_t^{-1}$ 代价极高（$O(d^3)$，$d$ 是参数维度）。

**（3）曲率感知的近似**

CTR-LoRA 使用对角 Hessian 近似或 Fisher 信息矩阵近似：

$$
H_t \approx \text{diag}(\mathbf{h}_t)
$$

其中 $\mathbf{h}_t$ 是每个参数维度上的曲率估计。对于 LoRA 参数 $A$ 和 $B$，曲率估计可以通过梯度的平方的滑动平均来计算：

$$
\mathbf{h}_t = \gamma \mathbf{h}_{t-1} + (1 - \gamma) (\nabla \mathcal{L}_t)^2
$$

其中 $\gamma$ 是衰减因子，通常取 0.9 或 0.99。这类似于 Adam 优化器中的二阶矩估计。

**（4）曲率感知的更新**

将曲率信息融入 LoRA 的更新中：

$$
\theta_{t+1} = \theta_t - \eta \frac{\nabla \mathcal{L}(\theta_t)}{\sqrt{\mathbf{h}_t} + \epsilon}
$$

这实际上是 Adam 优化器的形式。CTR-LoRA 的改进在于：对 LoRA 的 $A$ 和 $B$ 分别使用不同的曲率估计，因为 $A$ 和 $B$ 在数值尺度上差异很大（$B$ 初始化为 0，$A$ 随机初始化）。

### 6.1.3 信任域引导的数学推导

**（1）信任域的基本思想**

信任域方法的核心是：在每次更新时，约束参数变化不要太大。具体地，求解以下约束优化问题：

$$
\max_{\Delta\theta} \ \mathcal{L}(\theta_t + \Delta\theta) \quad \text{s.t.} \quad \|\Delta\theta\| \leq \delta
$$

其中 $\delta$ 是信任域半径。

**（2）拉格朗日对偶**

将约束优化转化为无约束优化，引入拉格朗日乘子 $\lambda$：

$$
\max_{\Delta\theta} \ \mathcal{L}(\theta_t + \Delta\theta) - \lambda (\|\Delta\theta\|^2 - \delta^2)
$$

代入二阶泰勒展开：

$$
\max_{\Delta\theta} \ \nabla \mathcal{L}^\top \Delta\theta + \frac{1}{2} \Delta\theta^\top H \Delta\theta - \lambda \|\Delta\theta\|^2
$$

对 $\Delta\theta$ 求导并令为 0：

$$
\nabla \mathcal{L} + H \Delta\theta - 2\lambda \Delta\theta = 0
$$

$$
\Delta\theta = -(H - 2\lambda I)^{-1} \nabla \mathcal{L}
$$

**（3）信任域半径的自适应调整**

CTR-LoRA 根据训练过程中的曲率变化动态调整信任域半径：

$$
\delta_t = \delta_0 \cdot \exp\left( -\alpha \cdot \frac{\|\nabla \mathcal{L}_t\|}{\|\nabla \mathcal{L}_0\|} \right)
$$

其中 $\delta_0$ 是初始半径，$\alpha$ 是衰减系数。梯度较大时，半径缩小，更新更保守；梯度较小时，半径增大，更新更激进。

**（4）秩调度**

CTR-LoRA 还引入秩调度机制：训练初期使用较小的秩（如 $r=4$），随着训练进行逐渐增大秩（如 $r=16$）。这基于以下观察：

- 训练初期，模型需要快速适应新任务，低秩足以捕捉主要方向。
- 训练后期，模型需要精细调整，高秩提供更强的表达能力。

秩调度策略：

$$
r_t = r_{\min} + (r_{\max} - r_{\min}) \cdot \min\left( 1, \frac{t}{T_{\text{schedule}}} \right)
$$

其中 $r_{\min}$ 和 $r_{\max}$ 分别是初始秩和最终秩，$T_{\text{schedule}}$ 是调度周期。

### 6.1.4 与标准 LoRA 的对比

| 维度         | LoRA         | CTR-LoRA             |
| ------------ | ------------ | -------------------- |
| 优化器       | Adam         | 曲率感知的自适应优化 |
| 更新幅度控制 | 固定学习率   | 信任域动态调整       |
| 秩           | 固定         | 动态调度             |
| 训练稳定性   | 对学习率敏感 | 更稳定               |
| 表达能力     | 受固定秩约束 | 动态秩增强           |
| 实现复杂度   | 低           | 中等                 |

### 6.1.5 代码示例

```python
import torch
import torch.nn as nn
import math

class CTRLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r_init=4, r_max=16,
                 lora_alpha=32, trust_radius=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_init = r_init
        self.r_max = r_max
        self.r_current = r_init
        self.lora_alpha = lora_alpha
        self.trust_radius = trust_radius

        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        # LoRA 旁路（初始为最大秩，通过掩码控制有效秩）
        self.lora_A = nn.Parameter(torch.zeros(r_max, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r_max))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # 曲率估计（二阶矩）
        self.register_buffer('h_A', torch.zeros_like(self.lora_A))
        self.register_buffer('h_B', torch.zeros_like(self.lora_B))

        self.scaling = lora_alpha / r_max

    def set_rank(self, r):
        """设置当前有效秩"""
        self.r_current = min(r, self.r_max)
        # 将超出有效秩的部分置零
        with torch.no_grad():
            self.lora_A[self.r_current:].zero_()
            self.lora_B[:, self.r_current:].zero_()

    def forward(self, x):
        # 只使用有效秩的部分
        A = self.lora_A[:self.r_current]
        B = self.lora_B[:, :self.r_current]
        original = nn.functional.linear(x, self.weight)
        lora_out = x @ A.T @ B.T
        return original + lora_out * self.scaling

    def update_curvature(self, grad_A, grad_B, gamma=0.99):
        """更新曲率估计"""
        with torch.no_grad():
            self.h_A = gamma * self.h_A + (1 - gamma) * grad_A ** 2
            self.h_B = gamma * self.h_B + (1 - gamma) * grad_B ** 2
```

## 6.2 TS²（Training with Sparsemax+, Testing with Softmax）

### 6.2.1 核心问题：SFT 导致的输出多样性下降

标准 SFT 使用交叉熵损失（等价于 Softmax + NLL），训练目标是最小化：

$$
\mathcal{L}_{\text{CE}} = -\sum_{t=1}^{T} \log \frac{\exp(z_{y_t})}{\sum_{v=1}^{V} \exp(z_v)}
$$

其中 $z_v$ 是词汇表中第 $v$ 个词的 logit，$y_t$ 是真实标签。

**问题**：Softmax 的输出永远不会精确为 0，即模型对所有词都分配了非零概率。当模型在训练集上过度优化时，输出分布会变得越来越尖锐（sharp），但同时也变得过度自信，导致：

- **输出多样性下降**：模型倾向于生成训练集中出现过的“安全”回答，缺乏创造性。
- **过拟合**：模型对训练分布的拟合过于紧密，泛化能力下降。

### 6.2.2 Sparsemax 的数学定义

Sparsemax 是 Softmax 的稀疏替代方案，由 Martins 和 Astudillo 于 2016 年提出。其定义为：

$$
\text{sparsemax}(z)_i = \max(0, z_i - \tau(z))
$$

其中 $\tau(z)$ 是阈值函数，定义为：

$$
\tau(z) = \frac{\sum_{j \in S(z)} z_j - 1}{|S(z)|}
$$

$S(z)$ 是满足 $z_j > \tau(z)$ 的索引集合。

**直观理解**：Sparsemax 将小于阈值的 logit 直接置零，输出是稀疏的概率分布。这意味着模型会将概率质量集中在少数几个词上，而其他词的概率精确为 0。

**Sparsemax 的性质**：

- 输出是稀疏的（很多元素为 0）
- 输出仍然在单纯形上（$\sum_i \text{sparsemax}(z)_i = 1$，$\text{sparsemax}(z)_i \geq 0$）
- 是投影到单纯形上的欧氏距离最近点

**Sparsemax 的梯度**：

$$
\frac{\partial \text{sparsemax}(z)_i}{\partial z_j} = \frac{1}{|S(z)|} \left( \delta_{ij} - \mathbb{1}[i \in S(z), j \in S(z)] \right)
$$

其中 $\delta_{ij}$ 是 Kronecker delta。

### 6.2.3 Sparsemax+ 的改进

标准 Sparsemax 存在一个问题：当所有 logit 都很接近时，输出可能过于稀疏，导致模型学习信号不足。Sparsemax+ 引入了一个平滑参数 $\epsilon$：

$$
\text{sparsemax}^+(z)_i = \max(0, z_i - \tau^+(z))
$$

其中阈值 $\tau^+(z)$ 满足：

$$
\sum_i \max(0, z_i - \tau^+(z)) = 1 - \epsilon
$$

即 Sparsemax+ 允许输出分布的总和为 $1 - \epsilon$（而不是 1），剩余的 $\epsilon$ 概率质量均匀分布在所有词上。这提供了最低限度的平滑，避免了过度稀疏。

### 6.2.4 TS² 的训练与推理

**训练阶段**：使用 Sparsemax+ 损失：

$$
\mathcal{L}_{\text{TS}^2} = -\sum_{t=1}^{T} \log \text{sparsemax}^+(z_{y_t})
$$

其中 $z_{y_t}$ 是真实标签对应的 logit。

**推理阶段**：使用标准 Softmax：

$$
P(y_t = v) = \frac{\exp(z_v / T)}{\sum_{u=1}^{V} \exp(z_u / T)}
$$

其中 $T$ 是温度参数。

**为什么训练和推理使用不同函数？**

- **训练时**：Sparsemax+ 的稀疏梯度使模型更关注少数关键 token，减少对无关词的梯度信号，提升训练效率。同时，稀疏性抑制了模型对训练分布的过度拟合。
- **推理时**：Softmax 生成更平滑的分布，保留了更多候选词，提升了输出的多样性。

### 6.2.5 数学推导：Sparsemax 的投影形式

Sparsemax 可以看作是将 logit 向量投影到概率单纯形上的欧氏投影：

$$
\text{sparsemax}(z) = \arg\min_{p \in \Delta^{V-1}} \|p - z\|^2
$$

其中 $\Delta^{V-1} = \{p \in \mathbb{R}^V : \sum_i p_i = 1, p_i \geq 0\}$ 是概率单纯形。

**推导**：

拉格朗日函数为：

$$
\mathcal{L}(p, \mu, \lambda) = \frac{1}{2}\|p - z\|^2 - \mu \left( \sum_i p_i - 1 \right) - \sum_i \lambda_i p_i
$$

对 $p_i$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial p_i} = p_i - z_i - \mu - \lambda_i = 0
$$

由 KKT 条件，$\lambda_i \geq 0$，$\lambda_i p_i = 0$。当 $p_i > 0$ 时，$\lambda_i = 0$，所以 $p_i = z_i + \mu$。当 $p_i = 0$ 时，$z_i + \mu \leq 0$。

因此，$p_i = \max(0, z_i + \mu)$，其中 $\mu$ 由约束 $\sum_i p_i = 1$ 确定。令 $\tau = -\mu$，即得 sparsemax 的形式。

### 6.2.6 效果对比

实验表明，TS² 在准确性和输出多样性上均优于标准 SFT：

| 指标       | 标准 SFT | TS²        |
| ---------- | -------- | ---------- |
| 准确性     | 基准     | 相当或略优 |
| 输出多样性 | 低       | 显著提升   |
| 过拟合程度 | 高       | 低         |
| 训练效率   | 基准     | 略优       |

### 6.2.7 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def sparsemax(z, dim=-1):
    """Sparsemax 实现"""
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device).float()
    k = k.view([1] * (z.dim() - 1) + [-1])
    support = z_sorted > (z_cumsum - 1) / k
    support_size = support.sum(dim=dim, keepdim=True).float()
    tau = (z_cumsum.gather(dim, (support_size - 1).long().clamp(min=0)) - 1) / support_size
    return torch.clamp(z - tau, min=0)

def sparsemax_plus(z, epsilon=0.01, dim=-1):
    """Sparsemax+ 实现"""
    p_sparse = sparsemax(z, dim=dim)
    V = z.size(dim)
    p_smooth = (1 - epsilon) * p_sparse + epsilon / V
    return p_smooth

class TS2Loss(nn.Module):
    def __init__(self, epsilon=0.01):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # logits: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len]
        probs = sparsemax_plus(logits, self.epsilon)
        # 取出 target 对应的概率
        target_probs = probs.gather(
            dim=-1,
            index=targets.unsqueeze(-1)
        ).squeeze(-1)
        # 避免 log(0)
        loss = -torch.log(target_probs + 1e-10)
        # 忽略 padding（target = -100）
        mask = (targets != -100).float()
        loss = (loss * mask).sum() / mask.sum()
        return loss
```

## 6.3 TRAPO（Trust-Region Adaptive Policy Optimization）

### 6.3.1 核心思想

TRAPO（信任域自适应策略优化）针对 SFT 和 RL 训练目标不一致的问题，提出在每个训练实例中**交替进行 SFT 和 RL 优化**：

- 在**专家前缀**（expert prefix）上优化 SFT 损失（最大化专家 token 的概率）
- 在**模型自身生成**上优化 RL 损失（最大化奖励）

这种交替训练方式使模型既能学习专家的示范，又能通过 RL 探索超越专家。

### 6.3.2 数学推导

**（1）SFT 损失**

对于专家示范 $(x, y^*)$，SFT 损失为：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y^*_t \mid y^*_{<t}, x)
$$

**（2）RL 损失**

对于模型生成的输出 $y \sim \pi_\theta(\cdot \mid x)$，RL 损失（PPO 风格）为：

$$
\mathcal{L}_{\text{RL}}(\theta) = -\mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^{T} \min\left( \rho_t(\theta) A_t, \ \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

其中 $\rho_t(\theta) = \frac{\pi_\theta(y_t \mid y_{<t}, x)}{\pi_{\text{old}}(y_t \mid y_{<t}, x)}$。

**（3）TRAPO 的组合损失**

TRAPO 将两者组合，使用一个权重 $\lambda$ 平衡：

$$
\mathcal{L}_{\text{TRAPO}}(\theta) = \lambda \mathcal{L}_{\text{SFT}}(\theta) + (1 - \lambda) \mathcal{L}_{\text{RL}}(\theta)
$$

**关键创新**：TRAPO 不是在整个批次上简单地混合两个损失，而是在**每个训练实例**中交替使用：

- 对于专家示范的前缀部分，使用 SFT 损失
- 对于模型生成的续写部分，使用 RL 损失

具体地，对于每个训练样本，将其分为两部分：

- **专家前缀** $y^*_{1:k}$（由专家生成）
- **模型续写** $y_{k+1:T}$（由模型生成）

则 TRAPO 损失为：

$$
\mathcal{L}_{\text{TRAPO}}(\theta) = -\sum_{t=1}^{k} \log \pi_\theta(y^*_t \mid y^*_{<t}, x) - \sum_{t=k+1}^{T} \log \pi_\theta(y_t \mid y_{<t}, x) \cdot A_t
$$

### 6.3.3 信任域自适应

TRAPO 的"信任域自适应"指的是：根据训练阶段动态调整 SFT 和 RL 的权重。

训练初期：模型需要从专家示范中学习，SFT 权重较高：

$$
\lambda_t = \lambda_{\max} \cdot \exp\left( -\frac{t}{T_{\text{decay}}} \right) + \lambda_{\min}
$$

训练后期：模型已经掌握了基础知识，可以更多地进行 RL 探索：

$$
\lambda_t = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \cdot \exp\left( -\frac{t}{T_{\text{decay}}} \right)
$$

其中 $\lambda_{\max}$ 和 $\lambda_{\min}$ 是权重的上下界，$T_{\text{decay}}$ 是衰减周期。

### 6.3.4 与标准 SFT + RL 的对比

| 维度     | 标准 SFT → RL        | TRAPO            |
| -------- | -------------------- | ---------------- |
| 训练方式 | 先 SFT 再 RL，两阶段 | 交替进行，单阶段 |
| 数据使用 | SFT 数据 + RL 数据   | 混合数据         |
| 专家利用 | 仅在 SFT 阶段        | 训练全程         |
| 探索能力 | RL 阶段才探索        | 全程探索         |
| 训练效率 | 需要两轮训练         | 单轮完成         |
| 遗忘风险 | SFT 后 RL 可能遗忘   | 交替训练，遗忘少 |

### 6.3.5 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TRAPOTrainer:
    def __init__(self, model, ref_model, lambda_init=0.8, lambda_min=0.1,
                 lambda_decay=1000, epsilon=0.2):
        self.model = model
        self.ref_model = ref_model
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_decay = lambda_decay
        self.epsilon = epsilon
        self.step_count = 0

    def get_lambda(self):
        """动态调整 SFT 和 RL 的权重"""
        lam = self.lambda_min + (self.lambda_init - self.lambda_min) * \
              torch.exp(torch.tensor(-self.step_count / self.lambda_decay))
        return lam.item()

    def compute_loss(self, expert_inputs, expert_labels,
                     model_inputs, model_labels, rewards, expert_prefix_len):
        """
        expert_inputs: 专家示范的输入
        model_inputs: 模型生成的输入
        rewards: 模型生成输出的奖励
        expert_prefix_len: 专家前缀长度
        """
        lam = self.get_lambda()

        # SFT 损失（在专家前缀上）
        sft_logits = self.model(expert_inputs).logits
        sft_loss = F.cross_entropy(
            sft_logits[:, :expert_prefix_len].reshape(-1, sft_logits.size(-1)),
            expert_labels[:, :expert_prefix_len].reshape(-1),
            ignore_index=-100
        )

        # RL 损失（在模型生成上）
        with torch.no_grad():
            ref_logits = self.ref_model(model_inputs).logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        model_logits = self.model(model_inputs).logits
        model_log_probs = F.log_softmax(model_logits, dim=-1)

        # 计算重要性采样比率
        ratio = torch.exp(
            model_log_probs.gather(-1, model_labels.unsqueeze(-1)).squeeze(-1) -
            ref_log_probs.gather(-1, model_labels.unsqueeze(-1)).squeeze(-1)
        )

        # 优势函数（使用奖励）
        advantages = rewards.unsqueeze(-1).expand_as(ratio)

        # PPO 裁剪损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        rl_loss = -torch.min(surr1, surr2).mean()

        # 组合损失
        total_loss = lam * sft_loss + (1 - lam) * rl_loss

        self.step_count += 1
        return total_loss, sft_loss.item(), rl_loss.item()
```

## 6.4 MDPO（Multi-Dimensional Label Enhanced DPO）

### 6.4.1 核心思想

MDPO（多维度标签增强 DPO）针对多模态 LLM 微调中 DPO 偏好信号单一的问题，在传统 DPO 的基础上引入**多维度标签信号**。传统 DPO 仅使用二元偏好（好/差），而 MDPO 使用多个维度的评分（如准确性、流畅性、相关性、安全性等）来增强偏好信号。

### 6.4.2 传统 DPO 的局限

标准 DPO 损失为：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

**局限**：

1. **二元信号**：只区分"好"和"差"，丢失了偏好强度的信息。
2. **单一维度**：只考虑整体偏好，无法区分不同维度的表现（如一个回答准确性高但流畅性差）。
3. **多模态场景不足**：在多模态任务中，需要同时考虑文本质量和图像-文本对齐质量。

### 6.4.3 多维度标签的数学表达

假设有 $K$ 个评价维度，每个回答 $y$ 有一个 $K$ 维的评分向量：

$$
\mathbf{s}(x, y) = (s_1(x, y), s_2(x, y), \dots, s_K(x, y))
$$

其中 $s_k \in [0, 1]$ 表示第 $k$ 个维度的评分。

例如，对于多模态任务，维度可以是：

- $s_1$：文本准确性
- $s_2$：图像-文本对齐度
- $s_3$：语言流畅性
- $s_4$：安全性

### 6.4.4 MDPO 的损失函数推导

**（1）多维度 Bradley-Terry 模型**

将 Bradley-Terry 模型扩展到多维度：

$$
P(y_w \succ y_l \mid x) = \sigma\left( \sum_{k=1}^{K} w_k \left( s_k(x, y_w) - s_k(x, y_l) \right) \right)
$$

其中 $w_k$ 是第 $k$ 个维度的权重。

**（2）引入隐式奖励**

MDPO 将每个维度的评分与隐式奖励关联：

$$
r_k(x, y) = \beta_k \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \gamma_k
$$

其中 $\beta_k$ 和 $\gamma_k$ 是第 $k$ 个维度的参数。

**（3）MDPO 损失**

将多维度的隐式奖励代入 Bradley-Terry 模型，得到 MDPO 损失：

$$
\mathcal{L}_{\text{MDPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \sum_{k=1}^{K} w_k \left( \hat{r}_k(x, y_w) - \hat{r}_k(x, y_l) \right) \right) \right]
$$

其中：

$$
\hat{r}_k(x, y) = \beta_k \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

**（4）简化形式**

当所有维度的 $\beta_k$ 相同时（即 $\beta_k = \beta$），MDPO 损失简化为：

$$
\mathcal{L}_{\text{MDPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \sum_{k=1}^{K} w_k \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]
$$

由于 $\sum_k w_k$ 是常数，可以合并到 $\beta$ 中，因此当 $\beta_k$ 相同时，MDPO 退化为标准 DPO 的加权版本。

**（5）梯度分析**

对 MDPO 损失求梯度：

$$
\nabla_\theta \mathcal{L}_{\text{MDPO}} = -\beta \mathbb{E} \left[ \sigma\left( -\Delta \hat{r} \right) \sum_{k=1}^{K} w_k \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right) \right]
$$

其中 $\Delta \hat{r} = \sum_k w_k (\hat{r}_k(x, y_w) - \hat{r}_k(x, y_l))$。

### 6.4.5 权重 $w_k$ 的确定

权重 $w_k$ 可以通过以下方式确定：

1. **人工设定**：根据任务重要性手动设定。
2. **学习得到**：将 $w_k$ 作为可训练参数，通过元学习优化。
3. **基于方差**：$w_k \propto 1 / \text{Var}(s_k)$，即方差小的维度权重更高（更稳定）。

### 6.4.6 实验效果

在 LLaVA-1.5-7B 和 Qwen-VL-Chat 上的实验表明，MDPO 相比传统 DPO：

- 训练损失降低 0.6%-6.6%
- 在多模态基准（MMBench、MM-Vet、POPE）上提升 1-3 个百分点
- 在多维度评估中，各维度均有提升

### 6.4.7 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDPOLoss(nn.Module):
    def __init__(self, beta=0.1, dim_weights=None):
        super().__init__()
        self.beta = beta
        # dim_weights: 各维度的权重，如果为 None 则均匀
        self.dim_weights = dim_weights

    def forward(self, policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                dim_scores_chosen=None, dim_scores_rejected=None):
        """
        policy_chosen_logps: [batch] 策略模型对 chosen 的 log 概率
        policy_rejected_logps: [batch] 策略模型对 rejected 的 log 概率
        ref_chosen_logps: [batch] 参考模型对 chosen 的 log 概率
        ref_rejected_logps: [batch] 参考模型对 rejected 的 log 概率
        dim_scores_chosen: [batch, K] chosen 的各维度评分
        dim_scores_rejected: [batch, K] rejected 的各维度评分
        """
        # 计算隐式奖励
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)

        # 多维度加权
        if dim_scores_chosen is not None and dim_scores_rejected is not None:
            # 使用多维度评分调整奖励
            dim_diff = dim_scores_chosen - dim_scores_rejected  # [batch, K]
            if self.dim_weights is not None:
                weighted_diff = (dim_diff * self.dim_weights).sum(dim=-1)
            else:
                weighted_diff = dim_diff.mean(dim=-1)
            logits = chosen_rewards - rejected_rewards + weighted_diff
        else:
            logits = chosen_rewards - rejected_rewards

        loss = -F.logsigmoid(logits).mean()
        return loss
```

## 6.5 样本效率缩放定律

### 6.5.1 核心发现

最新的样本效率缩放定律研究（TACL 2026）揭示了 PEFT 与全参微调的样本效率存在**架构依赖性差异**：

- **Encoder-only 模型**（如 BERT、RoBERTa）：多种 PEFT 方法在约 700 条样本以下能匹配或超越全参微调。
- **Decoder-only 模型**（如 LLaMA、GPT）：全参微调在所有观测样本量下都优于 LoRA。

### 6.5.2 实验设置

研究者在多个数据集（GLUE、SuperGLUE、指令微调数据集）上，使用多个模型（BERT、RoBERTa、LLaMA-2、Mistral）进行了系统实验，控制变量包括：

- 训练样本量：从 100 到 100,000
- 微调方法：全参微调、LoRA、Adapter、Prefix Tuning
- 模型规模：从 100M 到 13B

### 6.5.3 数学建模

研究者提出了一个样本效率缩放模型：

$$
\text{Performance}(N) = P_{\infty} - A \cdot N^{-\alpha}
$$

其中：

- $N$：训练样本量
- $P_{\infty}$：无限数据下的渐近性能
- $A$：缩放系数
- $\alpha$：缩放指数

对于全参微调和 PEFT，参数 $A$ 和 $\alpha$ 不同：

- **全参微调**：$\alpha_{\text{FFT}} \approx 0.3$
- **LoRA**：$\alpha_{\text{LoRA}} \approx 0.2$

这意味着全参微调随数据量增长的效率更高。

**交叉点分析**：

设全参微调和 PEFT 的性能曲线为：

$$
P_{\text{FFT}}(N) = P_{\infty}^{\text{FFT}} - A_{\text{FFT}} \cdot N^{-\alpha_{\text{FFT}}}
$$

$$
P_{\text{PEFT}}(N) = P_{\infty}^{\text{PEFT}} - A_{\text{PEFT}} \cdot N^{-\alpha_{\text{PEFT}}}
$$

令两者相等，求解交叉点 $N^*$：

$$
P_{\infty}^{\text{FFT}} - A_{\text{FFT}} \cdot N^{-\alpha_{\text{FFT}}} = P_{\infty}^{\text{PEFT}} - A_{\text{PEFT}} \cdot N^{-\alpha_{\text{PEFT}}}
$$

对于 encoder-only 模型，$N^* \approx 700$；对于 decoder-only 模型，$N^* \to \infty$（即全参微调始终更优）。

### 6.5.4 为什么存在架构差异

**（1）Encoder-only 模型的特点**

- 参数量较小（通常 110M-340M）
- 双向注意力机制
- 预训练任务（MLM）与下游任务（分类、NER）较为接近

这些特点使得 PEFT 的低秩约束足以捕捉任务相关的更新。

**（2）Decoder-only 模型的特点**

- 参数量巨大（7B-70B+）
- 因果注意力机制
- 预训练任务（CLM）与下游任务（指令遵循、推理）差异较大

这些特点使得 PEFT 的低秩约束不足以捕捉任务相关的全部更新，全参微调的优势更明显。

### 6.5.5 实践指导

基于上述发现，实践建议为：

| 场景                          | 推荐方法             | 理由                 |
| ----------------------------- | -------------------- | -------------------- |
| Encoder 模型 + 小样本（<700） | PEFT（BitFit、LoRA） | 效率高，性能相当     |
| Encoder 模型 + 大样本（>700） | 全参微调             | 性能更优             |
| Decoder 模型 + 小样本         | LoRA/QLoRA           | 资源约束下的实用选择 |
| Decoder 模型 + 大样本         | 全参微调             | 性能最优             |
| Decoder 模型 + 资源极度受限   | QLoRA                | 唯一可行方案         |

## 6.6 与其他技术的联合优化

### 6.6.1 压缩与微调的联合

**（1）核心思想**

模型压缩（剪枝、量化）与微调可以整合为统一的流水线：先压缩再微调，或边压缩边微调。这种联合优化可以降低计算成本、缩短训练时间，同时保持任务性能。

**（2）数学建模**

设原始模型参数为 $\theta$，压缩后的模型参数为 $\theta_c = \mathcal{C}(\theta)$，其中 $\mathcal{C}$ 是压缩算子。联合优化的目标为：

$$
\min_{\theta_c} \ \mathcal{L}_{\text{task}}(\theta_c) + \lambda \mathcal{R}(\theta_c)
$$

其中 $\mathcal{R}(\theta_c)$ 是压缩正则项（如稀疏性约束），$\lambda$ 是平衡系数。

**（3）典型流程**

```
预训练模型 → 压缩（剪枝/量化）→ 微调 → 部署
                ↓
         压缩感知的微调
```

**（4）压缩感知微调的损失**

在微调过程中，同时优化任务损失和压缩约束：

$$
\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{task}} + \lambda \sum_{l} \|W_l\|_1
$$

其中 $\|W_l\|_1$ 是第 $l$ 层权重的 L1 范数，促进稀疏性。

### 6.6.2 知识蒸馏与微调的结合

**（1）核心思想**

知识蒸馏（Knowledge Distillation, KD）可以将大模型（教师）的知识迁移到小模型（学生）。与微调结合时，典型的流程是：

1. 微调一个大模型作为教师。
2. 使用教师模型的输出（软标签）训练一个学生模型。
3. 学生模型可以进一步微调以适应特定任务。

**（2）KD 损失函数**

标准 KD 损失由两部分组成：

$$
\mathcal{L}_{\text{KD}} = \alpha \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + (1 - \alpha) \tau^2 \mathcal{L}_{\text{KL}}\left( \sigma\left( \frac{z_t}{\tau} \right) \parallel \sigma\left( \frac{z_s}{\tau} \right) \right)
$$

其中：

- $z_s$：学生模型的 logits
- $z_t$：教师模型的 logits
- $\sigma$：Softmax 函数
- $\tau$：温度参数，控制软标签的平滑程度
- $\alpha$：平衡系数

**（3）温度参数的推导**

温度参数 $\tau$ 的作用是平滑概率分布。当 $\tau \to \infty$ 时，分布趋于均匀；当 $\tau \to 0$ 时，分布趋于 one-hot。

对 logits $z$ 使用温度 $\tau$ 的 Softmax：

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

温度越高，分布越平滑，学生模型可以学到更多"暗知识"（dark knowledge），即非目标类别的概率信息。

**（4）与微调的结合流程**

```
步骤1：微调教师模型（如 LLaMA-3-70B）在目标任务上
步骤2：使用教师模型生成软标签（或直接使用 logits）
步骤3：训练学生模型（如 LLaMA-3-8B）：
    - 任务损失（硬标签）
    - 蒸馏损失（软标签）
步骤4：可选：对学生模型进行进一步微调
```

**（5）代码示例**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        """
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        """
        # 硬标签损失
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # 软标签损失（KL 散度）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            student_soft, teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 组合损失
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        return total_loss, ce_loss.item(), kl_loss.item()
```

### 6.6.3 多任务微调

**（1）核心思想**

多任务微调（Multi-task Fine-tuning）是在多个相关任务上同时微调模型，使模型能够跨任务共享知识。相比单任务微调，多任务微调通常能提升泛化能力。

**（2）损失函数**

多任务损失是各任务损失的加权和：

$$
\mathcal{L}_{\text{multi}} = \sum_{k=1}^{K} \lambda_k \mathcal{L}_k(\theta)
$$

其中 $\lambda_k$ 是第 $k$ 个任务的权重。

**（3）任务权重的确定**

- **均匀权重**：$\lambda_k = 1/K$
- **基于数据量**：$\lambda_k \propto N_k$（$N_k$ 是第 $k$ 个任务的样本量）
- **基于不确定性**：使用 Kendall 等人的方法，将权重作为可学习参数：

$$
\mathcal{L}_{\text{multi}} = \sum_{k=1}^{K} \frac{1}{2\sigma_k^2} \mathcal{L}_k(\theta) + \log \sigma_k
$$

其中 $\sigma_k$ 是第 $k$ 个任务的可学习不确定性参数。

**推导**：假设每个任务的损失服从高斯分布，似然为：

$$
p(\mathcal{L}_k \mid \theta) = \mathcal{N}(\mathcal{L}_k; \mu_k, \sigma_k^2)
$$

最大化对数似然等价于最小化：

$$
-\log p = \frac{1}{2\sigma_k^2} (\mathcal{L}_k - \mu_k)^2 + \log \sigma_k
$$

忽略常数项 $\mu_k$，即得到上述多任务损失。

### 6.6.4 持续学习与微调

**（1）核心问题**

持续学习（Continual Learning）关注模型在序列任务上训练时不遗忘旧任务。与微调结合时，核心挑战是：在微调新任务时，如何保留旧任务的知识。

**（2）EWC（Elastic Weight Consolidation）**

EWC 通过约束重要参数的更新来缓解遗忘：

$$
\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2
$$

其中：

- $\theta^*$：旧任务训练后的参数
- $F_i$：Fisher 信息矩阵的对角元素，衡量参数 $i$ 对旧任务的重要性
- $\lambda$：正则化强度

**Fisher 信息的计算**：

$$
F_i = \mathbb{E}_{x \sim \mathcal{D}_{\text{old}}} \left[ \left( \frac{\partial \log p(y \mid x, \theta)}{\partial \theta_i} \right)^2 \right]
$$

**（3）与 LoRA 的结合**

EWC 可以与 LoRA 结合：对 LoRA 参数施加 EWC 约束，保留旧任务的知识：

$$
\mathcal{L} = \mathcal{L}_{\text{new}} + \frac{\lambda}{2} \sum_i F_i^{(\text{LoRA})} (\phi_i - \phi_i^*)^2
$$

其中 $\phi$ 是 LoRA 参数，$F^{(\text{LoRA})}$ 是 LoRA 参数的 Fisher 信息。

### 6.6.5 联邦微调

**（1）核心思想**

联邦微调（Federated Fine-tuning）在多个客户端上分布式地微调模型，各客户端的数据不离开本地，只交换模型更新。这保护了数据隐私。

**（2）FedAvg 算法**

联邦平均（FedAvg）是最基本的联邦学习算法：

1. 服务器将全局模型 $\theta$ 发送给各客户端。
2. 每个客户端 $k$ 在本地数据上训练，得到更新 $\theta_k$。
3. 服务器聚合各客户端的更新：

$$
\theta_{\text{new}} = \sum_{k=1}^{K} \frac{N_k}{N} \theta_k
$$

其中 $N_k$ 是客户端 $k$ 的样本量，$N = \sum_k N_k$。

**（3）与 LoRA 的结合**

联邦微调可以与 LoRA 结合，进一步减少通信开销：各客户端只交换 LoRA 参数（很小），而不是完整模型。

**（4）挑战**

- **数据异质性**：各客户端数据分布不同，直接平均可能导致性能下降。
- **通信效率**：需要多轮通信。
- **隐私保护**：梯度可能泄露隐私，需要差分隐私等技术。



# 七、微调实战工具链

## 7.1 Hugging Face PEFT

### 7.1.1 核心功能

Hugging Face PEFT（Parameter-Efficient Fine-Tuning）是一个专注于参数高效微调的开源库，与 Transformers 库无缝集成。其核心功能包括：

- **多种 PEFT 方法**：支持 LoRA、QLoRA、DoRA、Adapter、Prefix Tuning、Prompt Tuning、IA³、BitFit、AdaLoRA 等。
- **统一接口**：所有方法都通过 `LoraConfig`、`PrefixTuningConfig` 等配置类定义，使用 `get_peft_model` 包装原始模型。
- **适配器管理**：支持适配器的保存、加载、合并和切换，便于多任务部署。
- **量化集成**：与 `bitsandbytes` 集成，支持 4bit/8bit 量化加载。
- **训练友好**：与 `Trainer`、`SFTTrainer` 等训练器兼容。

PEFT 的核心设计哲学是：**冻结预训练模型的大部分参数，仅训练少量新增参数**，从而在保持性能的同时大幅降低计算和存储成本。

### 7.1.2 基本用法

使用 PEFT 进行 LoRA 微调的基本流程：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载预训练模型和分词器
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 定义 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型
    r=16,                          # 秩
    lora_alpha=32,                 # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
    lora_dropout=0.05,             # Dropout
    bias="none"                    # 偏置处理
)

# 3. 包装模型
peft_model = get_peft_model(model, lora_config)

# 4. 查看可训练参数
peft_model.print_trainable_parameters()
# 输出示例: trainable params: 6,815,744 || all params: 8,037,404,672 || trainable%: 0.0848

# 5. 训练（使用 Trainer 或自定义训练循环）
# ...

# 6. 保存适配器
peft_model.save_pretrained("./lora_adapter")

# 7. 加载适配器
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name)
loaded_model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 8. 合并适配器到基础模型（推理时无额外延迟）
merged_model = loaded_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

### 7.1.3 常用 API 参数表

#### `LoraConfig` 参数表

| 参数名           | 含义             | 推荐值         | 说明                          |
| ---------------- | ---------------- | -------------- | ----------------------------- |
| `r`              | 低秩矩阵的秩     | 8-64           | 越大表达能力越强，参数量增加  |
| `lora_alpha`     | 缩放因子         | 16-64          | 通常设为 r 的 2 倍            |
| `target_modules` | 应用 LoRA 的模块 | q_proj, v_proj | 可扩展到所有线性层            |
| `lora_dropout`   | Dropout 概率     | 0.05-0.1       | 防止过拟合                    |
| `bias`           | 偏置处理方式     | "none"         | 可选 "all" 或 "lora_only"     |
| `task_type`      | 任务类型         | CAUSAL_LM      | 可选 SEQ_CLS、SEQ_2_SEQ_LM 等 |
| `use_dora`       | 是否使用 DoRA    | False          | 启用权重分解                  |
| `use_rslora`     | 是否使用 RS-LoRA | False          | 秩稳定缩放                    |

#### `get_peft_model` 参数表

| 参数名         | 含义       | 作用               |
| -------------- | ---------- | ------------------ |
| `model`        | 预训练模型 | 要包装的模型       |
| `peft_config`  | PEFT 配置  | 定义 PEFT 方法     |
| `adapter_name` | 适配器名称 | 多适配器时使用     |
| `mixed`        | 混合适配器 | 是否使用混合适配器 |

### 7.1.4 代码示例：完整 LoRA 微调

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载模型和分词器
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. 加载和预处理数据集
dataset = load_dataset("json", data_files="train.json", split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. 定义 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

# 4. 包装模型
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb"
)

# 6. 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 7. 创建训练器
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 8. 开始训练
trainer.train()

# 9. 保存适配器
peft_model.save_pretrained("./lora_adapter")
```

### 7.1.5 公式推导：LoRA 参数量与缩放因子

#### 7.1.5.1 参数量推导

原始权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 的参数量为：

$$
N_{\text{orig}} = d \times k
$$

LoRA 将更新分解为两个低秩矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$，参数量为：

$$
N_{\text{LoRA}} = d \times r + r \times k = r(d + k)
$$

参数减少比例：

$$
\frac{N_{\text{LoRA}}}{N_{\text{orig}}} = \frac{r(d + k)}{d \times k} = r \left( \frac{1}{k} + \frac{1}{d} \right)
$$

当 $d = k = 4096$，$r = 16$ 时：

$$
\frac{N_{\text{LoRA}}}{N_{\text{orig}}} = 16 \times \left( \frac{1}{4096} + \frac{1}{4096} \right) = \frac{32}{4096} = 0.0078 = 0.78\%
$$

即参数量仅为原来的 0.78%。

#### 7.1.5.2 缩放因子推导

LoRA 的更新量为：

$$
\Delta W = \frac{\alpha}{r} B A
$$

缩放因子 $\alpha/r$ 的作用是：当调整秩 $r$ 时，保持 LoRA 更新的有效幅度大致不变。

**推导**：假设 $A$ 的元素独立同分布，均值为 0，方差为 $\sigma_A^2$；$B$ 的元素独立同分布，均值为 0，方差为 $\sigma_B^2$。则 $B A$ 的每个元素的方差为：

$$
\text{Var}((BA)_{ij}) = \sum_{l=1}^{r} \text{Var}(B_{il} A_{lj}) = r \sigma_B^2 \sigma_A^2
$$

因此，$B A$ 的尺度随 $\sqrt{r}$ 增长。为了使其尺度与 $r$ 无关，需要除以 $r$，即使用 $\frac{1}{r} B A$。而 $\alpha$ 是一个超参数，用于控制整体缩放。因此，实际更新量为 $\frac{\alpha}{r} B A$。

当 $\alpha$ 固定时，不同 $r$ 下的更新幅度大致相当，便于超参数迁移。

## 7.2 TRL（Transformer Reinforcement Learning）

### 7.2.1 核心功能

TRL（Transformer Reinforcement Learning）是 Hugging Face 推出的强化学习训练库，专注于大语言模型的偏好优化与对齐。其核心功能包括：

- **SFT 训练**：`SFTTrainer` 用于监督微调。
- **偏好优化**：`DPOTrainer`、`GRPOTrainer`、`PPOTrainer`、`RewardTrainer` 等。
- **奖励模型训练**：`RewardTrainer` 用于训练奖励模型。
- **与 PEFT 集成**：支持 LoRA/QLoRA 与各种训练器组合。
- **与 Accelerate 集成**：支持分布式训练。

TRL 的核心设计目标是：**让偏好优化训练变得简单、稳定、可扩展**。

### 7.2.2 支持的训练器

| 训练器          | 用途                | 核心算法           |
| --------------- | ------------------- | ------------------ |
| `SFTTrainer`    | 监督微调            | 交叉熵损失         |
| `DPOTrainer`    | 直接偏好优化        | DPO 损失           |
| `GRPOTrainer`   | 组相对策略优化      | GRPO 损失          |
| `PPOTrainer`    | 近端策略优化        | PPO 损失           |
| `RewardTrainer` | 奖励模型训练        | Bradley-Terry 损失 |
| `ORPOTrainer`   | odds ratio 偏好优化 | ORPO 损失          |
| `KTOTrainer`    | KTO 偏好优化        | KTO 损失           |

### 7.2.3 常用 API 参数表

#### `DPOConfig` 参数表

| 参数名                        | 含义         | 推荐值       | 说明                   |
| ----------------------------- | ------------ | ------------ | ---------------------- |
| `beta`                        | KL 惩罚系数  | 0.1-0.5      | 控制偏离参考模型的程度 |
| `learning_rate`               | 学习率       | 5e-7 至 5e-6 | 比 SFT 更小            |
| `num_train_epochs`            | 训练轮数     | 1-3          | 通常 1 轮即可          |
| `per_device_train_batch_size` | 批次大小     | 1-4          | 根据显存调整           |
| `gradient_accumulation_steps` | 梯度累积     | 4-16         | 模拟大批次             |
| `max_length`                  | 最大长度     | 1024         | 序列最大长度           |
| `max_prompt_length`           | 提示最大长度 | 512          | 提示部分最大长度       |

#### `GRPOConfig` 参数表

| 参数名                  | 含义         | 推荐值 | 说明                 |
| ----------------------- | ------------ | ------ | -------------------- |
| `num_generations`       | 组大小 G     | 4-16   | 每个提示采样的输出数 |
| `beta`                  | KL 惩罚系数  | 0.04   | 通常比 DPO 小        |
| `learning_rate`         | 学习率       | 1e-6   | 比 SFT 小            |
| `max_completion_length` | 最大生成长度 | 256    | 输出最大长度         |

### 7.2.4 代码示例：DPO 训练

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# 1. 加载模型和分词器
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载偏好数据集
dataset = load_dataset("json", data_files="dpo_data.json", split="train")

# 3. DPO 配置
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 4. 创建训练器
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 5. 开始训练
trainer.train()
```

### 7.2.5 公式推导：DPO 损失与梯度

#### 7.2.5.1 DPO 损失推导

DPO 的出发点是最优策略的闭式解：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$。

反解奖励：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

代入 Bradley-Terry 模型：

$$
P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

得到：

$$
P(y_w \succ y_l \mid x) = \sigma\left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
$$

用 $\pi_\theta$ 替代 $\pi^*$，最大化偏好似然，等价于最小化负对数似然：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

#### 7.2.5.2 DPO 梯度推导

令：

$$
z = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

损失为 $\mathcal{L} = -\log \sigma(z)$。

对 $z$ 求导：

$$
\frac{\partial \mathcal{L}}{\partial z} = -\frac{\sigma'(z)}{\sigma(z)} = -(1 - \sigma(z)) = \sigma(z) - 1
$$

而：

$$
\frac{\partial z}{\partial \theta} = \beta \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right)
$$

因此：

$$
\nabla_\theta \mathcal{L} = (\sigma(z) - 1) \cdot \beta \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right)
$$

注意到 $\sigma(z) - 1 = -\sigma(-z)$，所以：

$$
\nabla_\theta \mathcal{L} = -\beta \sigma(-z) \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right)
$$

当模型错误地偏好 $y_l$ 时（即 $z < 0$），$\sigma(-z)$ 较大，梯度会推动 $y_w$ 的概率增大，$y_l$ 的概率减小。当模型正确偏好 $y_w$ 时（$z > 0$），$\sigma(-z)$ 趋近于 0，梯度趋近于 0。

## 7.3 Unsloth

### 7.3.1 核心功能

Unsloth 是一个专注于**加速大模型微调**的开源库，由 Daniel Han 和 Michael Han 开发。其核心功能包括：

- **速度提升**：比标准 Hugging Face 实现快 2-5 倍。
- **显存优化**：减少 50%-80% 的显存占用。
- **支持 LoRA/QLoRA**：与 PEFT 兼容。
- **支持多种模型**：LLaMA、Mistral、Qwen、Gemma 等。
- **易于使用**：仅需替换几行代码即可加速。

Unsloth 的核心优化技术包括：

- **自定义 Triton 内核**：手写 GPU 内核优化计算。
- **梯度检查点优化**：更高效的梯度检查点实现。
- **内存管理优化**：减少中间激活值的存储。
- **FlashAttention 集成**：使用 FlashAttention 加速注意力计算。

### 7.3.2 优化技术

#### 7.3.2.1 自定义 Triton 内核

Triton 是 OpenAI 开发的 GPU 编程语言，Unsloth 使用 Triton 手写了多个关键计算的内核，包括：

- **RoPE 旋转位置编码**：优化的旋转计算。
- **RMSNorm**：优化的层归一化。
- **交叉熵损失**：优化的损失计算。
- **注意力计算**：优化的注意力内核。

这些内核减少了内存访问次数，提高了计算效率。

#### 7.3.2.2 梯度检查点优化

梯度检查点（Gradient Checkpointing）是一种用计算换显存的技术：在前向传播时不保存所有中间激活值，而是在反向传播时重新计算。Unsloth 的优化版本进一步减少了重计算的开销。

#### 7.3.2.3 FlashAttention 集成

FlashAttention 是一种高效的注意力计算算法，通过分块计算和重计算减少显存访问。Unsloth 集成了 FlashAttention-2，进一步加速训练。

### 7.3.3 代码示例

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 加载模型（Unsloth 自动优化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3-8B",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # 自动检测
)

# 2. 添加 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth 优化版
)

# 3. 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        bf16=True,
        output_dir="./unsloth_output",
    ),
)

trainer.train()
```

### 7.3.4 公式推导：FlashAttention 的显存复杂度

标准注意力计算的显存复杂度为 $O(N^2)$，其中 $N$ 是序列长度。FlashAttention 通过分块计算将显存复杂度降低到 $O(N)$。

**标准注意力**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d}} \right) V
$$

需要存储 $N \times N$ 的注意力矩阵，显存为 $O(N^2)$。

**FlashAttention**：

将 $Q, K, V$ 分块，逐块计算注意力，避免存储完整的 $N \times N$ 矩阵。具体地，将序列分成大小为 $B$ 的块，显存复杂度为：

$$
O\left( \frac{N}{B} \times B^2 \right) = O(NB)
$$

当 $B$ 固定时，显存为 $O(N)$。这使得长序列训练成为可能。

## 7.4 LLaMA-Factory

### 7.4.1 核心功能

LLaMA-Factory 是一个**集成化的大模型微调框架**，由 HiYouga 等人开发。其核心功能包括：

- **支持 100+ 模型**：LLaMA、Mistral、Qwen、ChatGLM、Baichuan 等。
- **多种微调方法**：全参微调、LoRA、QLoRA、DoRA、DPO、GRPO 等。
- **Web UI**：提供可视化界面，无需编写代码即可训练。
- **命令行接口**：支持脚本化训练。
- **数据管理**：内置多种数据格式和预处理工具。
- **模型导出**：支持合并 LoRA、量化导出。

### 7.4.2 支持的模型和方法

#### 支持的模型（部分）

| 模型系列 | 代表模型                 |
| -------- | ------------------------ |
| LLaMA    | LLaMA-3-8B, LLaMA-3-70B  |
| Mistral  | Mistral-7B, Mixtral-8x7B |
| Qwen     | Qwen2.5-7B, Qwen2.5-72B  |
| ChatGLM  | ChatGLM3-6B              |
| Baichuan | Baichuan2-7B             |
| Yi       | Yi-6B, Yi-34B            |

#### 支持的微调方法

| 方法     | 说明            |
| -------- | --------------- |
| 全参微调 | 更新所有参数    |
| LoRA     | 低秩适应        |
| QLoRA    | 量化 + LoRA     |
| DoRA     | 权重分解 + LoRA |
| AdaLoRA  | 动态秩分配      |
| DPO      | 直接偏好优化    |
| GRPO     | 组相对策略优化  |
| PPO      | 近端策略优化    |

### 7.4.3 代码示例

#### 命令行训练

```bash
llamafactory-cli train \
    --stage sft \
    --model_name_or_path meta-llama/Llama-3-8B \
    --dataset my_dataset \
    --template llama3 \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --output_dir ./sft_lora \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --logging_steps 10 \
    --save_steps 500
```

#### Web UI 训练

```bash
llamafactory-cli webui
```

然后在浏览器中打开 `http://localhost:7860`，配置模型、数据集和超参数，点击“开始训练”。

### 7.4.4 配置参数表

#### 数据集配置（`dataset_info.json`）

```json
{
  "my_dataset": {
    "file_name": "train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }
}
```

#### 训练参数表

| 参数名                        | 含义       | 推荐值 | 说明                   |
| ----------------------------- | ---------- | ------ | ---------------------- |
| `stage`                       | 训练阶段   | sft    | 可选 sft、dpo、ppo 等  |
| `finetuning_type`             | 微调方法   | lora   | 可选 full、lora、qlora |
| `lora_rank`                   | LoRA 秩    | 16     | 同 PEFT 的 r           |
| `lora_alpha`                  | LoRA 缩放  | 32     | 同 PEFT 的 alpha       |
| `learning_rate`               | 学习率     | 2e-5   | 根据任务调整           |
| `num_train_epochs`            | 训练轮数   | 3      | 小数据集可增加         |
| `per_device_train_batch_size` | 批次大小   | 4      | 根据显存调整           |
| `gradient_accumulation_steps` | 梯度累积   | 8      | 有效批次 = 4×8=32      |
| `lr_scheduler_type`           | 学习率调度 | cosine | 余弦退火               |
| `warmup_ratio`                | 预热比例   | 0.1    | 前 10% 步数预热        |
| `bf16`                        | 混合精度   | true   | 推荐使用 bf16          |

## 7.5 vLLM

### 7.5.1 核心功能

vLLM 是一个专注于**大模型推理加速**的开源库，由 UC Berkeley 开发。其核心功能包括：

- **PagedAttention**：高效的内存管理，大幅提升吞吐量。
- **连续批处理（Continuous Batching）**：动态调度请求，提高 GPU 利用率。
- **高吞吐量**：比 Hugging Face 默认实现快 14-24 倍。
- **兼容 OpenAI API**：可直接替换 OpenAI API。
- **支持量化**：GPTQ、AWQ、FP8 等。
- **分布式推理**：支持张量并行和流水线并行。

### 7.5.2 PagedAttention 原理

#### 7.5.2.1 问题背景

在标准注意力计算中，每个序列的 KV Cache（键值缓存）需要连续的内存空间。当序列长度变化时，内存分配和释放会产生碎片，导致显存利用率低下。此外，不同序列的 KV Cache 大小不同，难以共享。

#### 7.5.2.2 PagedAttention 的核心思想

PagedAttention 借鉴了操作系统的虚拟内存分页机制，将 KV Cache 分成固定大小的块（block），每个块可以存储在非连续的内存中。通过页表（page table）将逻辑块映射到物理块。

**数学表达**：

对于序列 $i$，其 KV Cache 被分成 $B_i$ 个块，每个块大小为 $P$（通常 $P=16$）。第 $j$ 个块存储了第 $jP$ 到 $(j+1)P-1$ 个 token 的键值。

注意力计算时，通过页表查找每个逻辑块对应的物理块：

$$
\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left( \frac{Q_i K_i^\top}{\sqrt{d}} \right) V_i
$$

其中 $K_i$ 和 $V_i$ 由多个块拼接而成：

$$
K_i = [K_i^{(1)}; K_i^{(2)}; \dots; K_i^{(B_i)}]
$$

#### 7.5.2.3 显存节省

PagedAttention 将 KV Cache 的内存浪费从 60%-80% 降低到不足 4%，使得相同显存下可以服务更多请求。

### 7.5.3 代码示例

```python
from vllm import LLM, SamplingParams

# 1. 加载模型
llm = LLM(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=1,  # 张量并行数
    dtype="bfloat16",
    gpu_memory_utilization=0.9,  # GPU 显存利用率
)

# 2. 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

# 3. 推理
prompts = [
    "什么是机器学习？",
    "请解释一下 Transformer 架构。",
]
outputs = llm.generate(prompts, sampling_params)

# 4. 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}\n")
```

### 7.5.4 公式推导：吞吐量与延迟

#### 7.5.4.1 吞吐量定义

吞吐量（Throughput）定义为单位时间内处理的 token 数：

$$
\text{Throughput} = \frac{\text{Total Tokens}}{\text{Total Time}}
$$

#### 7.5.4.2 连续批处理

在连续批处理中，请求动态加入和离开批次。假设有 $N$ 个请求，每个请求生成长度为 $L_i$ 的序列，总时间为 $T$，则吞吐量为：

$$
\text{Throughput} = \frac{\sum_{i=1}^{N} L_i}{T}
$$

#### 7.5.4.3 延迟

首 token 延迟（Time to First Token, TTFT）和每 token 延迟（Time per Output Token, TPOT）：

$$
\text{TTFT} = t_{\text{first}}
$$

$$
\text{TPOT} = \frac{T - t_{\text{first}}}{\sum_i L_i - N}
$$

#### 7.5.4.4 PagedAttention 对吞吐量的提升

标准注意力中，KV Cache 的显存浪费导致可同时处理的请求数减少。PagedAttention 将显存浪费从 $W$ 降低到 $W'$（$W' \ll W$），则可同时处理的请求数从 $N$ 增加到 $N'$：

$$
N' = N \cdot \frac{1 - W'}{1 - W}
$$

例如，$W = 0.6$，$W' = 0.04$，则：

$$
N' = N \cdot \frac{0.96}{0.4} = 2.4N
$$

即吞吐量提升约 2.4 倍。

## 7.6 工具链对比

### 7.6.1 对比表格

| 工具          | 定位         | 核心优势                       | 适用阶段 | 学习曲线 |
| ------------- | ------------ | ------------------------------ | -------- | -------- |
| PEFT          | 参数高效微调 | 方法最全，与 Transformers 集成 | 微调     | 低       |
| TRL           | 对齐训练     | SFT/DPO/GRPO 一体化            | 对齐     | 中       |
| Unsloth       | 加速微调     | 速度快，显存低                 | 微调     | 低       |
| LLaMA-Factory | 集成框架     | Web UI，多模型支持             | 全流程   | 低       |
| vLLM          | 推理服务     | 高吞吐，低延迟                 | 部署     | 中       |

### 7.6.2 选型建议

#### 7.6.2.1 按任务阶段选择

- **微调阶段**：PEFT（灵活）或 Unsloth（加速）或 LLaMA-Factory（一站式）。
- **对齐阶段**：TRL（DPO/GRPO/PPO）。
- **部署阶段**：vLLM（高吞吐）。

#### 7.6.2.2 按资源条件选择

| 资源条件                | 推荐工具组合              |
| ----------------------- | ------------------------- |
| 单卡 24GB，微调 7B      | Unsloth + QLoRA           |
| 多卡 A100，微调 13B-70B | LLaMA-Factory + LoRA/全参 |
| 需要快速实验            | Unsloth + TRL             |
| 需要 Web UI             | LLaMA-Factory             |
| 生产部署                | vLLM                      |

#### 7.6.2.3 按团队能力选择

- **研究团队**：PEFT + TRL，灵活控制每个细节。
- **工程团队**：LLaMA-Factory + vLLM，快速上线。
- **个人开发者**：Unsloth + PEFT，资源友好。

#### 7.6.2.4 典型工作流

```
数据准备 → LLaMA-Factory（SFT + LoRA）
         → TRL（DPO/GRPO 对齐）
         → PEFT（合并适配器）
         → vLLM（部署推理）
```

或者：

```
数据准备 → Unsloth（加速 LoRA 微调）
         → PEFT（保存适配器）
         → vLLM（部署推理）
```

### 7.6.3 工具链集成示例

```python
# 1. 使用 Unsloth 加速 LoRA 微调
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3-8B",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32)

# 2. 使用 TRL 进行 DPO 对齐
from trl import DPOTrainer, DPOConfig
trainer = DPOTrainer(
    model=model,
    args=DPOConfig(output_dir="./dpo_output", beta=0.1),
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# 3. 使用 PEFT 合并适配器
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
peft_model = PeftModel.from_pretrained(base_model, "./dpo_output")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# 4. 使用 vLLM 部署
from vllm import LLM
llm = LLM(model="./merged_model", tensor_parallel_size=1)
outputs = llm.generate(["什么是机器学习？"], SamplingParams(max_tokens=256))
print(outputs[0].outputs[0].text)
```



# 八、微调中的关键问题

## 8.1 灾难性遗忘（Catastrophic Forgetting）

### 8.1.1 问题描述

灾难性遗忘是指：模型在微调新任务后，在旧任务或通用能力上的表现显著下降。这是神经网络的一个根本性问题，最早由 McCloskey 和 Cohen 于 1989 年在认知科学领域提出。

**直观理解**：

想象一个模型在预训练阶段学会了“翻译英语”、“写代码”、“回答常识问题”等多种能力。当我们在一个特定的客服问答数据集上微调它时，模型可能会“忘记”如何写代码，甚至忘记如何翻译——这就是灾难性遗忘。

**具体表现**：

- 微调后模型在目标任务上表现优秀，但在通用基准（如 MMLU、HellaSwag）上大幅下降。
- 模型可能遗忘预训练阶段学到的语法规则，输出不连贯的文本。
- 在多语言模型上，微调一种语言可能损害其他语言的能力。

### 8.1.2 数学原理

**（1）参数更新的冲突**

设预训练模型参数为 $\theta_{\text{pre}}$，旧任务的损失函数为 $\mathcal{L}_{\text{old}}$，新任务的损失函数为 $\mathcal{L}_{\text{new}}$。

微调时，我们只优化新任务：

$$
\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{new}}(\theta)
$$

但希望 $\theta^*$ 在旧任务上也表现良好：

$$
\mathcal{L}_{\text{old}}(\theta^*) \approx \mathcal{L}_{\text{old}}(\theta_{\text{pre}})
$$

**问题**：$\mathcal{L}_{\text{new}}$ 和 $\mathcal{L}_{\text{old}}$ 的最优参数可能位于参数空间的不同区域。优化 $\mathcal{L}_{\text{new}}$ 可能将参数移动到 $\mathcal{L}_{\text{old}}$ 的“坏”区域。

**（2）梯度干扰分析**

考虑参数 $\theta_i$ 在两个任务上的梯度：

$$
g_i^{\text{new}} = \frac{\partial \mathcal{L}_{\text{new}}}{\partial \theta_i}, \quad g_i^{\text{old}} = \frac{\partial \mathcal{L}_{\text{old}}}{\partial \theta_i}
$$

如果 $g_i^{\text{new}} \cdot g_i^{\text{old}} < 0$，则两个任务的梯度方向相反，更新 $\theta_i$ 以优化新任务会损害旧任务。

**（3）损失函数曲面的几何视角**

在参数空间中，旧任务的最优解 $\theta_{\text{old}}^*$ 和新任务的最优解 $\theta_{\text{new}}^*$ 通常是不同的点。微调过程是从 $\theta_{\text{pre}}$ 出发，沿着 $\mathcal{L}_{\text{new}}$ 的梯度方向移动到 $\theta_{\text{new}}^*$ 的过程。如果 $\theta_{\text{new}}^*$ 距离 $\theta_{\text{old}}^*$ 很远，则旧任务的损失在 $\theta_{\text{new}}^*$ 处会显著上升。

**（4）量化分析**

设参数更新量为 $\Delta\theta = \theta^* - \theta_{\text{pre}}$，旧任务损失的变化可以用二阶泰勒展开近似：

$$
\mathcal{L}_{\text{old}}(\theta_{\text{pre}} + \Delta\theta) \approx \mathcal{L}_{\text{old}}(\theta_{\text{pre}}) + \nabla \mathcal{L}_{\text{old}}(\theta_{\text{pre}})^\top \Delta\theta + \frac{1}{2} \Delta\theta^\top H_{\text{old}} \Delta\theta
$$

其中 $H_{\text{old}} = \nabla^2 \mathcal{L}_{\text{old}}(\theta_{\text{pre}})$ 是旧任务的 Hessian 矩阵。

如果 $\Delta\theta$ 在 $H_{\text{old}}$ 的大特征值方向上分量较大，则 $\mathcal{L}_{\text{old}}$ 上升明显。这说明：**参数更新在旧任务重要方向上的分量是遗忘的主要原因**。

### 8.1.3 缓解策略

#### 8.1.3.1 使用 PEFT 方法

**核心思想**：冻结大部分预训练参数，仅更新少量新增参数。这样预训练学到的知识大部分保留在冻结参数中。

**数学原理**：

设预训练参数为 $\theta_{\text{pre}}$，新增的可训练参数为 $\phi$。PEFT 的更新为：

$$
\theta^* = \theta_{\text{pre}} + \Delta\theta(\phi)
$$

其中 $\Delta\theta(\phi)$ 是由少量参数生成的更新量。例如 LoRA 中：

$$
\Delta W = \frac{\alpha}{r} B A
$$

由于 $\|\Delta\theta(\phi)\|$ 远小于全参微调的更新量，对旧任务的影响更小。

**效果**：LoRA 相比全参微调，在通用基准上的下降通常小 5-15 个百分点。

#### 8.1.3.2 混合通用数据

**核心思想**：在训练数据中混合一定比例的通用数据，让模型在适应新任务的同时保持通用能力。

**损失函数**：

设新任务数据集为 $\mathcal{D}_{\text{new}}$，通用数据集为 $\mathcal{D}_{\text{general}}$，则混合损失为：

$$
\mathcal{L}_{\text{mix}} = (1 - \lambda) \mathcal{L}_{\text{new}}(\theta; \mathcal{D}_{\text{new}}) + \lambda \mathcal{L}_{\text{general}}(\theta; \mathcal{D}_{\text{general}})
$$

其中 $\lambda$ 是混合比例。

**推荐比例**：

- 60%-80% 领域数据 + 20%-40% 通用数据
- 数据量越大、分布越窄的任务，需要更高比例的通用数据

**代码示例**：

```python
from datasets import concatenate_datasets, load_dataset

# 加载领域数据和通用数据
domain_dataset = load_dataset("json", data_files="domain_train.json", split="train")
general_dataset = load_dataset("json", data_files="general_train.json", split="train")

# 按比例采样
domain_size = len(domain_dataset)
general_size = int(domain_size * 0.3)  # 30% 通用数据
general_sampled = general_dataset.shuffle(seed=42).select(range(general_size))

# 合并
mixed_dataset = concatenate_datasets([domain_dataset, general_sampled])
mixed_dataset = mixed_dataset.shuffle(seed=42)
```

#### 8.1.3.3 EWC（Elastic Weight Consolidation）

**核心思想**：对旧任务重要的参数施加更强的约束，使其不易被改变。

**损失函数**：

$$
\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2
$$

其中：

- $\theta_i^*$：旧任务训练后的参数
- $F_i$：Fisher 信息矩阵的对角元素，衡量参数 $i$ 对旧任务的重要性
- $\lambda$：正则化强度

**Fisher 信息的推导**：

Fisher 信息矩阵定义为：

$$
F = \mathbb{E}_{x \sim p(x)} \left[ \nabla_\theta \log p(y \mid x, \theta) \nabla_\theta \log p(y \mid x, \theta)^\top \right]
$$

其对角元素为：

$$
F_i = \mathbb{E}_{x \sim \mathcal{D}_{\text{old}}} \left[ \left( \frac{\partial \log p(y \mid x, \theta)}{\partial \theta_i} \right)^2 \right]
$$

**直观理解**：$F_i$ 越大，说明参数 $\theta_i$ 对旧任务的输出影响越大，应该被保护。$F_i$ 越小，说明该参数对旧任务不重要，可以自由调整。

**EWC 的拉格朗日推导**：

考虑约束优化问题：

$$
\min_\theta \mathcal{L}_{\text{new}}(\theta) \quad \text{s.t.} \quad \mathcal{L}_{\text{old}}(\theta) \leq \mathcal{L}_{\text{old}}(\theta^*)
$$

将 $\mathcal{L}_{\text{old}}$ 在 $\theta^*$ 处二阶展开：

$$
\mathcal{L}_{\text{old}}(\theta) \approx \mathcal{L}_{\text{old}}(\theta^*) + \frac{1}{2} (\theta - \theta^*)^\top F (\theta - \theta^*)
$$

使用拉格朗日乘子法：

$$
\mathcal{L} = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} (\theta - \theta^*)^\top F (\theta - \theta^*)
$$

取对角近似，即得 EWC 损失。

**代码示例**：

```python
import torch
import torch.nn as nn

class EWCRegularizer:
    def __init__(self, model, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.old_params = {}

    def compute_fisher(self, dataset, num_samples=500):
        """计算 Fisher 信息矩阵的对角元素"""
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()

        for i, batch in enumerate(dataset):
            if i >= num_samples:
                break
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2

        # 归一化
        for n in self.fisher:
            self.fisher[n] /= num_samples

        # 保存旧参数
        self.old_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}

    def penalty(self):
        """计算 EWC 正则项"""
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.old_params[n]) ** 2).sum()
        return self.lambda_ewc * loss
```

#### 8.1.3.4 L2-LoRA（层特定 L2 正则化）

**核心思想**：在 LoRA 微调中，对 LoRA 权重施加 L2 正则化，约束其变化幅度。

**损失函数**：

$$
\mathcal{L}_{\text{L2-LoRA}} = \mathcal{L}_{\text{task}}(\theta_{\text{LoRA}}) + \lambda \sum_{l} \|B_l A_l\|_F^2
$$

其中 $\|\cdot\|_F$ 是 Frobenius 范数，$\lambda$ 是正则化强度。

**关键创新**：对不同层使用不同的正则化强度：

$$
\mathcal{L}_{\text{L2-LoRA}} = \mathcal{L}_{\text{task}}(\theta_{\text{LoRA}}) + \sum_{l} \lambda_l \|B_l A_l\|_F^2
$$

其中 $\lambda_l$ 是第 $l$ 层的正则化强度。通常对底层（靠近输入）使用更强的正则化，因为底层学到的特征更通用。

**代码示例**：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, config)

# 在训练循环中添加 L2 正则
def compute_loss_with_l2(model, batch, lambda_l2=0.01):
    outputs = model(**batch)
    task_loss = outputs.loss

    # 计算 LoRA 权重的 L2 正则
    l2_reg = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            l2_reg += param.pow(2).sum()

    total_loss = task_loss + lambda_l2 * l2_reg
    return total_loss
```

#### 8.1.3.5 经验回放（Experience Replay）

**核心思想**：在微调新任务时，混入一些旧任务的样本，让模型“复习”旧知识。

**方法**：

- 保存一部分旧任务样本（如 1%-5%）
- 在每个训练批次中，按比例混合新旧样本
- 损失函数为加权组合

**代码示例**：

```python
from torch.utils.data import ConcatDataset, DataLoader

# 旧任务数据（保留一小部分）
old_dataset = load_dataset("json", data_files="old_task_sample.json", split="train")

# 新任务数据
new_dataset = load_dataset("json", data_files="new_task.json", split="train")

# 合并数据集
combined_dataset = ConcatDataset([old_dataset, new_dataset])
dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
```

#### 8.1.3.6 策略对比

| 策略         | 原理           | 优点       | 缺点               | 适用场景       |
| ------------ | -------------- | ---------- | ------------------ | -------------- |
| PEFT         | 冻结大部分参数 | 简单、高效 | 表达能力受限       | 通用场景       |
| 混合通用数据 | 混入旧任务数据 | 效果好     | 需要额外数据       | 有通用数据     |
| EWC          | 约束重要参数   | 无需旧数据 | 计算 Fisher 开销大 | 无法访问旧数据 |
| L2-LoRA      | 约束 LoRA 权重 | 简单、有效 | 需要调 λ           | LoRA 微调      |
| 经验回放     | 混入旧样本     | 效果好     | 需要存储旧数据     | 有存储空间     |

## 8.2 数据质量问题

### 8.2.1 数据质量的维度

数据质量是微调成功的决定性因素。低质量数据会导致模型学到错误的模式，甚至产生有害输出。数据质量可以从以下维度衡量：

**（1）标注准确性**

输出是否正确。错误的标注会直接损害模型性能。研究表明，5%-10% 的标注错误率会显著降低模型性能。

**（2）覆盖度**

是否覆盖了目标场景。例如，一个客服数据集如果只有“退货”相关的样本，模型将无法处理“换货”或“投诉”问题。

**（3）一致性**

输出风格是否统一。如果同一类问题的回答风格不一致（有的正式、有的口语化），模型会学到混乱的输出模式。

**（4）平衡性**

各类样本比例是否合理。如果某类样本占比超过 90%，模型会偏向该类输出，忽略少数类。

**（5）充分性**

每类任务是否有足够的样本量。数据量决定了微调方法的上限。

### 8.2.2 数据质量评估的数学指标

**（1）标注一致性（Inter-Annotator Agreement）**

如果多个标注者对同一样本进行标注，可以用 Cohen's Kappa 系数衡量一致性：

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

其中：

- $p_o$：实际观察到的标注一致率
- $p_e$：随机一致率

**推导**：

$p_o$ 是标注者实际一致的样本比例：

$$
p_o = \frac{\text{一致的样本数}}{\text{总样本数}}
$$

$p_e$ 是如果标注者随机标注时的期望一致率：

$$
p_e = \sum_{k} p_{1,k} \cdot p_{2,k}
$$

其中 $p_{1,k}$ 是标注者 1 标注为类别 $k$ 的比例，$p_{2,k}$ 是标注者 2 标注为类别 $k$ 的比例。

**Kappa 系数的解释**：

- $\kappa > 0.8$：几乎完全一致
- $0.6 < \kappa \leq 0.8$：高度一致
- $0.4 < \kappa \leq 0.6$：中等一致
- $\kappa \leq 0.4$：一致性较差

**（2）数据多样性**

可以用独特 n-gram 比例衡量：

$$
\text{Diversity} = \frac{|\text{unique n-grams}|}{|\text{total n-grams}|}
$$

值越接近 1，多样性越高。

**（3）标签分布熵**

用信息熵衡量标签分布的均衡性：

$$
H = -\sum_{k=1}^{K} p_k \log p_k
$$

其中 $p_k$ 是第 $k$ 类样本的比例。熵越大，分布越均衡。

**（4）数据质量综合评分**

可以将多个维度加权组合：

$$
Q = w_1 \cdot \text{Accuracy} + w_2 \cdot \text{Coverage} + w_3 \cdot \text{Consistency} + w_4 \cdot \text{Balance}
$$

其中 $w_i$ 是各维度的权重，$\sum w_i = 1$。

### 8.2.3 数据质量提升方法

#### 8.2.3.1 使用更强模型进行数据清洗

**流程**：

1. 用强 LLM（如 GPT-4）对训练样本进行评分。
2. 过滤掉评分低于阈值的样本。
3. 对评分中等的样本进行人工审核。

**代码示例**：

```python
import openai

def score_sample(instruction, output):
    """用 GPT-4 对样本质量打分"""
    prompt = f"""
    请评估以下问答对的质量，从 1-5 分打分（5 分最好）。

    问题：{instruction}
    回答：{output}

    评估维度：
    1. 回答是否准确
    2. 回答是否完整
    3. 语言是否流畅
    4. 格式是否规范

    请输出分数和理由。
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def filter_dataset(dataset, threshold=4):
    """过滤低质量样本"""
    filtered = []
    for sample in dataset:
        score = score_sample(sample["instruction"], sample["output"])
        if float(score.split()[0]) >= threshold:
            filtered.append(sample)
    return filtered
```

#### 8.2.3.2 人工审核关键样本

对于关键应用，人工审核是必要的。审核策略包括：

- **全量审核**：数据量小（<1000 条）时，全部人工审核。
- **抽样审核**：数据量大时，按比例抽样（如 10%）。
- **重点审核**：对边界情况、高风险类别重点审核。

#### 8.2.3.3 数据增强

**（1）同义改写**

使用 LLM 将同一个问题改写为多种表达：

```python
def paraphrase(text, n=3):
    """生成多个改写版本"""
    prompt = f"请将以下问题改写为 {n} 个不同表达方式，保持语义不变：\n{text}"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.split("\n")
```

**（2）回译（Back Translation）**

将文本翻译成另一种语言，再翻译回来，得到语义相近但表达不同的文本：

```python
def back_translate(text, src_lang="zh", mid_lang="en"):
    """回译数据增强"""
    # 翻译到中间语言
    translated = translate(text, src_lang, mid_lang)
    # 翻译回原语言
    back_translated = translate(translated, mid_lang, src_lang)
    return back_translated
```

**（3）格式变换**

改变输出的格式（如从段落改为列表），保持内容不变。

#### 8.2.3.4 迭代式数据收集

**流程**：

1. 用当前模型在验证集上推理。
2. 收集模型出错的样本。
3. 人工标注这些样本。
4. 将新标注的样本加入训练集。
5. 重新训练模型。

**优势**：针对模型的薄弱环节补充数据，效率最高。

### 8.2.4 数据质量的常见陷阱

| 陷阱       | 表现                   | 后果             | 避免方法       |
| ---------- | ---------------------- | ---------------- | -------------- |
| 标注错误   | 输出与问题不匹配       | 模型学到错误模式 | 多轮审核       |
| 数据泄露   | 测试集样本出现在训练集 | 评估结果虚高     | 严格划分数据集 |
| 分布偏差   | 某类样本过多           | 模型偏向该类     | 分层采样       |
| 格式不一致 | 输出风格不统一         | 模型输出混乱     | 统一格式模板   |
| 长度偏差   | 训练数据回答都很长     | 模型输出过长     | 均衡长度分布   |
| 语言污染   | 混合多种语言           | 模型输出语言混乱 | 统一语言       |

## 8.3 过拟合问题

### 8.3.1 问题描述

过拟合是指模型在训练集上表现优异，但在验证集和新数据上表现下降。在微调中，过拟合尤其常见，因为微调数据通常较少，而模型参数量巨大。

**表现**：

- 训练损失持续下降，验证损失先下降后上升。
- 模型在训练集上的输出几乎完美，但在新样本上表现不稳定。
- 模型开始“记忆”训练数据的具体内容，而非学习通用模式。

### 8.3.2 数学原理

**（1）偏差-方差分解**

模型的期望泛化误差可以分解为：

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

其中：

- $\text{Bias}$：模型预测与真实值的系统性偏差
- $\text{Variance}$：模型对不同训练集的敏感度
- $\text{Noise}$：数据本身的不可约误差

**推导**：

设真实函数为 $f(x)$，模型预测为 $\hat{f}(x)$，观测值为 $y = f(x) + \epsilon$，其中 $\epsilon$ 是均值为 0、方差为 $\sigma^2$ 的噪声。

期望泛化误差为：

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \mathbb{E}[(f(x) + \epsilon - \hat{f}(x))^2]
$$

展开：

$$
= \mathbb{E}[(f(x) - \hat{f}(x))^2] + 2\mathbb{E}[(f(x) - \hat{f}(x))\epsilon] + \mathbb{E}[\epsilon^2]
$$

由于 $\epsilon$ 与 $\hat{f}(x)$ 独立且 $\mathbb{E}[\epsilon] = 0$，中间项为 0：

$$
= \mathbb{E}[(f(x) - \hat{f}(x))^2] + \sigma^2
$$

进一步分解第一项：

$$
\mathbb{E}[(f(x) - \hat{f}(x))^2] = \mathbb{E}[(\mathbb{E}[\hat{f}(x)] - f(x) + \hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
$$

$$
= (\mathbb{E}[\hat{f}(x)] - f(x))^2 + \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
$$

$$
= \text{Bias}^2 + \text{Variance}
$$

因此：

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2
$$

**过拟合的本质**：模型过于复杂，方差过大。

**（2）正则化的数学原理**

L2 正则化（权重衰减）的损失函数为：

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{task}}(\theta) + \frac{\lambda}{2} \|\theta\|^2
$$

**推导**：L2 正则化等价于对参数施加高斯先验。设参数 $\theta$ 的先验为 $\mathcal{N}(0, \sigma^2 I)$，则最大后验估计（MAP）为：

$$
\theta_{\text{MAP}} = \arg\max_\theta \left[ \log p(\mathcal{D} \mid \theta) + \log p(\theta) \right]
$$

其中 $\log p(\theta) = -\frac{1}{2\sigma^2} \|\theta\|^2 + C$。因此：

$$
\theta_{\text{MAP}} = \arg\min_\theta \left[ -\log p(\mathcal{D} \mid \theta) + \frac{1}{2\sigma^2} \|\theta\|^2 \right]
$$

即 L2 正则化的损失函数，其中 $\lambda = 1/\sigma^2$。

**（3）Dropout 的数学原理**

Dropout 在训练时随机将神经元的输出置零：

$$
h_i' = \frac{m_i}{1 - p} h_i
$$

其中 $m_i \sim \text{Bernoulli}(1 - p)$，$p$ 是丢弃概率。除以 $1 - p$ 是为了保持期望不变：

$$
\mathbb{E}[h_i'] = \mathbb{E}\left[ \frac{m_i}{1 - p} h_i \right] = \frac{\mathbb{E}[m_i]}{1 - p} h_i = \frac{1 - p}{1 - p} h_i = h_i
$$

**Dropout 防止过拟合的原理**：

- 训练时随机丢弃神经元，相当于训练了多个子网络的集成。
- 推理时使用完整网络，相当于对子网络进行平均。
- 这降低了模型对特定神经元的依赖，增强了泛化能力。

### 8.3.3 缓解策略

#### 8.3.3.1 早停（Early Stopping）

**核心思想**：在验证损失不再下降时停止训练。

**实现**：

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop
```

#### 8.3.3.2 增加 Dropout

在模型和 LoRA 中增加 Dropout：

```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,  # 增加 Dropout
    target_modules=["q_proj", "v_proj"],
)
```

#### 8.3.3.3 权重衰减

在优化器中设置权重衰减：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    weight_decay=0.01,  # L2 正则化
    ...
)
```

#### 8.3.3.4 数据增强

通过数据增强增加训练数据的多样性，减少过拟合风险。

#### 8.3.3.5 减少训练轮数

训练轮数过多是过拟合的主要原因之一。推荐：

- 小数据集（<1000 条）：1-2 轮
- 中等数据集（1000-10000 条）：2-3 轮
- 大数据集（>10000 条）：1-2 轮

#### 8.3.3.6 使用更小的学习率

较小的学习率减少参数更新幅度，降低过拟合风险。

#### 8.3.3.7 策略对比

| 策略     | 原理                   | 优点       | 缺点         |
| -------- | ---------------------- | ---------- | ------------ |
| 早停     | 验证损失不再下降时停止 | 简单、有效 | 需要验证集   |
| Dropout  | 随机丢弃神经元         | 集成效果   | 训练变慢     |
| 权重衰减 | L2 正则化              | 简单       | 需要调 λ     |
| 数据增强 | 增加数据多样性         | 效果好     | 需要额外处理 |
| 减少轮数 | 避免过度训练           | 简单       | 可能欠拟合   |
| 小学习率 | 减少更新幅度           | 稳定       | 训练慢       |

## 8.4 评估方法

### 8.4.1 自动评估指标

#### 8.4.1.1 分类任务

**准确率（Accuracy）** ：

$$
\text{Accuracy} = \frac{\text{正确预测数}}{\text{总样本数}}
$$

**精确率（Precision）** ：

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

**召回率（Recall）** ：

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

**F1 分数**：

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**AUC-ROC**：

ROC 曲线是以假阳性率（FPR）为横轴、真阳性率（TPR）为纵轴的曲线。AUC 是 ROC 曲线下的面积。

$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$

#### 8.4.1.2 生成任务

**BLEU 分数**：

BLEU 衡量生成文本与参考文本的 n-gram 重叠度：

$$
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)
$$

其中：

- $p_n$：n-gram 精确率
- $w_n$：权重，通常 $w_n = 1/N$
- BP：长度惩罚

$$
\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \leq r \end{cases}
$$

其中 $c$ 是生成文本长度，$r$ 是参考文本长度。

**ROUGE 分数**：

ROUGE-N 衡量 n-gram 召回率：

$$
\text{ROUGE-N} = \frac{\sum_{S \in \text{Ref}} \sum_{\text{n-gram} \in S} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{S \in \text{Ref}} \sum_{\text{n-gram} \in S} \text{Count}(\text{n-gram})}
$$

**BERTScore**：

使用 BERT 嵌入计算语义相似度：

$$
\text{BERTScore} = \frac{1}{|y|} \sum_{y_i \in y} \max_{\hat{y}_j \in \hat{y}} \text{cos}(\mathbf{e}_{y_i}, \mathbf{e}_{\hat{y}_j})
$$

#### 8.4.1.3 问答任务

**精确匹配（EM）** ：

$$
\text{EM} = \frac{\text{预测答案与标准答案完全匹配的样本数}}{\text{总样本数}}
$$

**F1 分数**：

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

其中 Precision 和 Recall 基于预测答案和标准答案的 token 重叠计算。

#### 8.4.1.4 推理任务

**准确率**：最终答案是否正确。

**通过率**：代码是否通过测试用例。

### 8.4.2 LLM-as-Judge

#### 8.4.2.1 核心思想

使用更强的 LLM（如 GPT-4、Claude）作为评判者，对模型输出进行质量评估。这适用于开放式生成任务，其中传统自动指标无法捕捉语义质量。

#### 8.4.2.2 实现方式

**（1）单答案评分**

```python
import openai

def llm_judge_single(question, answer):
    """对单个回答评分"""
    prompt = f"""
    请评估以下回答的质量，从 1-5 分打分（5 分最好）。

    问题：{question}
    回答：{answer}

    评估维度：
    1. 准确性：回答是否正确
    2. 相关性：回答是否与问题相关
    3. 流畅性：语言是否自然
    4. 完整性：是否覆盖所有要点

    请输出 JSON 格式：{{"score": <1-5>, "reason": "<理由>"}}
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    return response.choices[0].message.content
```

**（2）成对比较**

```python
def llm_judge_pair(question, answer_a, answer_b):
    """比较两个回答"""
    prompt = f"""
    请比较以下两个回答的质量。

    问题：{question}
    回答 A：{answer_a}
    回答 B：{answer_b}

    请输出 JSON 格式：{{"winner": "A" 或 "B", "reason": "<理由>"}}
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    return response.choices[0].message.content
```

#### 8.4.2.3 LLM-as-Judge 的偏差

| 偏差类型 | 表现                   | 缓解方法             |
| -------- | ---------------------- | -------------------- |
| 位置偏差 | 偏好第一个或第二个回答 | 交换顺序，取平均     |
| 长度偏差 | 偏好更长的回答         | 控制长度，或显式提示 |
| 自我偏好 | 偏好自己生成的回答     | 使用不同的模型评判   |
| 格式偏差 | 偏好特定格式的回答     | 统一格式             |

#### 8.4.2.4 减少偏差的方法

**（1）交换顺序**

对同一对回答，分别以 (A, B) 和 (B, A) 的顺序评判，取平均结果。

**（2）多模型评判**

使用多个不同的模型进行评判，取平均或投票。

**（3）显式约束**

在提示词中明确要求“不要因为长度而偏好”。

### 8.4.3 人工评估

#### 8.4.3.1 评估维度

| 维度       | 说明               | 评分标准 |
| ---------- | ------------------ | -------- |
| 准确性     | 回答是否正确       | 1-5 分   |
| 相关性     | 回答是否与问题相关 | 1-5 分   |
| 流畅性     | 语言是否自然流畅   | 1-5 分   |
| 安全性     | 是否包含有害内容   | 是/否    |
| 风格一致性 | 是否符合期望的风格 | 1-5 分   |
| 完整性     | 是否覆盖所有要点   | 1-5 分   |

#### 8.4.3.2 评估流程

```
1. 从测试集中随机抽取 N 个样本（N=100-500）
2. 让模型生成回答
3. 至少 3 位标注者独立评分
4. 计算评分者间一致性（Cohen's Kappa）
5. 对不一致的样本进行讨论和仲裁
6. 汇总评分结果
```

#### 8.4.3.3 人工评估的成本

| 项目             | 成本估算      |
| ---------------- | ------------- |
| 单条评估时间     | 1-3 分钟      |
| 单条评估费用     | 5-20 元       |
| 500 条评估总费用 | 2500-10000 元 |
| 评估周期         | 3-7 天        |

### 8.4.4 评估指标总结表

| 任务类型 | 自动指标               | LLM-as-Judge | 人工评估 |
| -------- | ---------------------- | ------------ | -------- |
| 分类     | 准确率、F1、AUC        | 可选         | 抽样验证 |
| 生成     | BLEU、ROUGE、BERTScore | 推荐         | 抽样验证 |
| 问答     | EM、F1                 | 推荐         | 抽样验证 |
| 推理     | 准确率、通过率         | 可选         | 抽样验证 |
| 对话     | 困惑度                 | 推荐         | 推荐     |
| 安全     | 有害内容检测           | 推荐         | 推荐     |

### 8.4.5 评估流程的最佳实践

#### 8.4.5.1 分层评估

```
第一层：自动指标（快速、低成本）
    ↓ 筛选
第二层：LLM-as-Judge（中等成本）
    ↓ 筛选
第三层：人工评估（高成本、高质量）
```

#### 8.4.5.2 持续评估

- **回归测试**：每次模型更新后，在固定的测试集上运行评估。
- **A/B 测试**：在生产环境中对比新旧模型的表现。
- **在线监控**：实时监控模型输出的质量指标。

#### 8.4.5.3 评估数据集构建

- **测试集与训练集严格分离**：不能有重叠。
- **测试集代表性**：覆盖所有重要场景。
- **测试集规模**：至少 200-500 条，确保统计显著性。
- **定期更新**：随着业务变化更新测试集。

#### 8.4.5.4 常见评估陷阱

| 陷阱         | 表现                   | 避免方法       |
| ------------ | ---------------------- | -------------- |
| 数据泄露     | 测试集样本出现在训练集 | 严格划分       |
| 过拟合测试集 | 反复在测试集上调参     | 使用验证集调参 |
| 指标单一     | 只看准确率             | 多指标综合评估 |
| 忽略长尾     | 只评估常见场景         | 分层评估       |
| 忽略安全     | 只评估性能             | 增加安全评估   |

### 8.4.6 代码示例：完整评估流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import openai

class ModelEvaluator:
    def __init__(self, model, tokenizer, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset

    def generate_predictions(self, max_new_tokens=256):
        """生成预测"""
        predictions = []
        for sample in self.test_dataset:
            inputs = self.tokenizer(sample["prompt"], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            prediction = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            predictions.append(prediction)
        return predictions

    def compute_automatic_metrics(self, predictions):
        """计算自动指标"""
        references = [s["reference"] for s in self.test_dataset]

        # BLEU
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )

        # ROUGE
        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(
            predictions=predictions,
            references=references
        )

        return {"bleu": bleu_score, "rouge": rouge_score}

    def llm_judge_evaluation(self, predictions, sample_size=100):
        """LLM-as-Judge 评估"""
        import random
        indices = random.sample(range(len(predictions)), min(sample_size, len(predictions)))

        scores = []
        for idx in indices:
            sample = self.test_dataset[idx]
            score = llm_judge_single(
                sample["prompt"],
                predictions[idx]
            )
            scores.append(score)

        avg_score = sum(s["score"] for s in scores) / len(scores)
        return {"average_score": avg_score, "detailed_scores": scores}

    def full_evaluation(self):
        """完整评估流程"""
        print("1. 生成预测...")
        predictions = self.generate_predictions()

        print("2. 计算自动指标...")
        auto_metrics = self.compute_automatic_metrics(predictions)

        print("3. LLM-as-Judge 评估...")
        llm_metrics = self.llm_judge_evaluation(predictions)

        print("4. 汇总结果...")
        results = {
            "automatic": auto_metrics,
            "llm_judge": llm_metrics,
        }
        return results
```



# 九、微调实战流程

## 9.1 需求分析与方法选择

### 9.1.1 明确微调目标

微调的第一步是明确目标：**我们到底要让模型学会什么？** 不同的目标对应不同的数据需求、方法选择和评估标准。

#### 9.1.1.1 微调目标的分类

| 目标类型     | 描述                                   | 典型场景                      | 数据需求        |
| ------------ | -------------------------------------- | ----------------------------- | --------------- |
| 格式适配     | 让模型按特定格式输出                   | 客服回复、JSON 输出、报告模板 | 500-2000 条     |
| 领域知识注入 | 让模型掌握领域术语和知识               | 医疗问答、法律咨询、金融分析  | 5000-20000 条   |
| 风格迁移     | 让模型模仿特定语气或风格               | 品牌调性、角色扮演、文学创作  | 1000-5000 条    |
| 推理能力增强 | 提升模型在数学、逻辑、代码上的推理能力 | 数学解题、代码生成、逻辑推理  | 20000-100000 条 |
| 安全对齐     | 让模型拒绝有害请求，输出安全内容       | 内容审核、价值观对齐          | 5000-20000 条   |

#### 9.1.1.2 目标的可量化定义

微调目标必须可量化，否则无法评估是否成功。例如：

- **格式适配**：输出符合目标格式的比例 ≥ 95%。
- **领域知识**：在领域测试集上的准确率 ≥ 85%。
- **风格迁移**：人工评估风格一致性平均分 ≥ 4.0/5.0。
- **推理增强**：在数学基准（如 GSM8K）上的准确率提升 ≥ 10 个百分点。
- **安全对齐**：有害请求的拒绝率 ≥ 99%。

#### 9.1.1.3 目标与成本的权衡

微调目标越高，所需的数据、计算资源和时间越多。例如：

- 仅格式适配：500 条数据 + 单卡 24GB GPU + 2 小时。
- 领域知识注入：10000 条数据 + 4×A100 + 1 天。
- 推理能力增强：50000 条数据 + 8×A100 + 3 天。

### 9.1.2 资源评估

在开始微调之前，必须评估可用的计算资源、数据资源和时间预算。

#### 9.1.2.1 计算资源评估

| 资源等级 | GPU 配置             | 可微调模型规模 | 推荐方法              |
| -------- | -------------------- | -------------- | --------------------- |
| 消费级   | RTX 3090/4090 (24GB) | 7B             | QLoRA                 |
| 专业级   | A100 40GB            | 7B-13B         | LoRA                  |
| 企业级   | A100/H100 80GB × 4   | 13B-70B        | LoRA / 全参（小模型） |
| 大型企业 | H100 80GB × 8+       | 70B+           | 全参 / LoRA           |

**显存需求估算公式**：

全参微调的显存需求为：

$$
\text{显存}_{\text{FFT}} \approx \underbrace{2P}_{\text{模型权重}} + \underbrace{2P}_{\text{梯度}} + \underbrace{8P}_{\text{优化器状态}} + \underbrace{A}_{\text{激活值}}
$$

其中 $P$ 是模型参数量（单位：十亿），$A$ 是激活值显存，与批次大小和序列长度相关。

对于 7B 模型：

$$
\text{显存}_{\text{FFT}} \approx 2 \times 7 + 2 \times 7 + 8 \times 7 + A = 14 + 14 + 56 + A = 84\text{GB} + A
$$

LoRA 微调的显存需求为：

$$
\text{显存}_{\text{LoRA}} \approx \underbrace{2P}_{\text{模型权重}} + \underbrace{2P_{\text{LoRA}}}_{\text{LoRA 梯度}} + \underbrace{8P_{\text{LoRA}}}_{\text{LoRA 优化器状态}} + \underbrace{A}_{\text{激活值}}
$$

其中 $P_{\text{LoRA}}$ 是 LoRA 参数量，通常为 $P$ 的 0.01%-1%。

对于 7B 模型，LoRA 参数量约 0.1%：

$$
\text{显存}_{\text{LoRA}} \approx 14 + 0.014 + 0.056 + A \approx 14\text{GB} + A
$$

QLoRA 进一步将模型权重量化为 4bit：

$$
\text{显存}_{\text{QLoRA}} \approx \underbrace{0.5P}_{\text{4bit 权重}} + \underbrace{2P_{\text{LoRA}}}_{\text{LoRA 梯度}} + \underbrace{8P_{\text{LoRA}}}_{\text{LoRA 优化器状态}} + \underbrace{A}_{\text{激活值}}
$$

对于 7B 模型：

$$
\text{显存}_{\text{QLoRA}} \approx 3.5 + 0.014 + 0.056 + A \approx 3.6\text{GB} + A
$$

#### 9.1.2.2 数据资源评估

| 数据量         | 推荐方法        | 训练轮数 | 预期效果                |
| -------------- | --------------- | -------- | ----------------------- |
| <500 条        | PEFT + 数据增强 | 3-5      | 可学会格式，难注入知识  |
| 500-2000 条    | LoRA            | 2-3      | 格式适配 + 部分领域知识 |
| 2000-10000 条  | LoRA            | 2-3      | 领域知识注入效果良好    |
| 10000-50000 条 | LoRA / 全参     | 1-2      | 复杂任务，推理增强      |
| >50000 条      | 全参微调        | 1-2      | 最佳效果                |

#### 9.1.2.3 时间预算评估

| 方法     | 7B 模型训练时间（单卡 A100） | 70B 模型训练时间（8×A100） |
| -------- | ---------------------------- | -------------------------- |
| QLoRA    | 3-4 小时                     | 12-24 小时                 |
| LoRA     | 2-3 小时                     | 8-12 小时                  |
| 全参微调 | 10-12 小时                   | 3-5 天                     |

### 9.1.3 方法选择决策树

根据目标、资源和数据量，可以使用以下决策树选择微调方法：

```
开始
  │
  ├── 数据量 > 10万条 + GPU（A100×8+）？
  │   ├── 是 → 全参微调（FFT）
  │   └── 否 → 继续判断
  │
  ├── 显存是否充足（>40GB）？
  │   ├── 是 → LoRA
  │   └── 否 → QLoRA
  │
  ├── 是否需要偏好对齐？
  │   ├── 是 → SFT + DPO/GRPO
  │   └── 否 → 仅 SFT
  │
  ├── 是否对推理延迟敏感？
  │   ├── 是 → LoRA（可合并权重）
  │   └── 否 → Adapter / Prefix Tuning
  │
  └── 是否追求低秩下的高精度？
      ├── 是 → DoRA
      └── 否 → LoRA
```

**决策示例**：

| 场景                                       | 推荐方法    | 理由                          |
| ------------------------------------------ | ----------- | ----------------------------- |
| 单卡 24GB，微调 7B 模型，数据 2000 条      | QLoRA + SFT | 显存受限，QLoRA 唯一可行      |
| 4×A100 40GB，微调 13B 模型，数据 50000 条  | LoRA + SFT  | 资源充足，LoRA 性价比最高     |
| 8×A100 80GB，微调 70B 模型，数据 100000 条 | 全参微调    | 数据充足，追求极致性能        |
| 单卡 24GB，微调 7B 模型，需要对齐          | QLoRA + DPO | 显存受限，DPO 比 PPO 更省显存 |

## 9.2 数据准备与处理

### 9.2.1 数据收集

#### 9.2.1.1 数据来源

| 来源       | 优点           | 缺点               | 成本        |
| ---------- | -------------- | ------------------ | ----------- |
| 人工标注   | 质量最高       | 成本高、速度慢     | 1-10 元/条  |
| 模型合成   | 速度快、成本低 | 可能引入错误       | 0.1-1 元/条 |
| 业务日志   | 真实、贴近场景 | 需要清洗、隐私问题 | 低          |
| 公开数据集 | 免费、质量较高 | 可能不匹配领域     | 免费        |
| 数据增强   | 扩展现有数据   | 可能引入噪声       | 低          |

#### 9.2.1.2 数据收集策略

**（1）冷启动阶段**

- 人工标注 200-500 条高质量种子数据。
- 使用强 LLM（如 GPT-4）基于种子数据生成更多样本。
- 人工审核生成的样本，过滤低质量数据。

**（2）迭代阶段**

- 部署初始模型，收集用户真实交互数据。
- 识别模型出错的样本，人工标注正确答案。
- 将新标注的样本加入训练集，重新训练。

**（3）数据规模估算**

根据任务复杂度，所需数据量可估算为：

$$
N_{\text{required}} = \frac{C}{\text{Information per sample}}
$$

其中 $C$ 是任务复杂度（如输出格式的约束数量、领域知识的广度），Information per sample 是每条样本提供的信息量。

实践经验：

- 输出格式约束：每增加一个约束，需要约 100-200 条样本。
- 领域知识：每覆盖一个子领域，需要约 500-1000 条样本。
- 推理能力：每提升 5% 准确率，需要约 5000-10000 条样本。

### 9.2.2 数据清洗与格式化

#### 9.2.2.1 数据清洗流程

```
原始数据
    ↓
去重（精确去重 + 语义去重）
    ↓
过滤低质量样本（长度过短、乱码、重复）
    ↓
PII 脱敏（去除或替换敏感信息）
    ↓
统一格式（标点、大小写、编码）
    ↓
人工审核（抽样）
    ↓
清洗后数据
```

**去重算法**：

精确去重用哈希：

$$
\text{hash}(s) = \text{MD5}(s)
$$

语义去重用嵌入相似度：

$$
\text{sim}(s_i, s_j) = \cos(\mathbf{e}_i, \mathbf{e}_j) > \tau
$$

其中 $\tau$ 是阈值（通常 0.95），$\mathbf{e}$ 是句子的嵌入向量。

**PII 脱敏示例**：

```python
import re

def desensitize(text):
    # 手机号
    text = re.sub(r'1[3-9]\d{9}', '<PHONE>', text)
    # 邮箱
    text = re.sub(r'\S+@\S+\.\S+', '<EMAIL>', text)
    # 身份证号
    text = re.sub(r'\d{17}[\dXx]', '<ID>', text)
    # 银行卡号
    text = re.sub(r'\d{16,19}', '<BANK_CARD>', text)
    return text
```

#### 9.2.2.2 数据格式化

**Alpaca 格式**：

```json
[
  {
    "instruction": "计算这些物品的总费用。",
    "input": "输入：汽车 - $3000，衣服 - $100，书 - $20。",
    "output": "汽车、衣服和书的总费用为 $3000 + $100 + $20 = $3120。"
  }
]
```

**ShareGPT 格式**：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好，请介绍一下自己。"},
      {"from": "gpt", "value": "你好！我是一个AI助手..."}
    ]
  }
]
```

**LLaMA-Factory 数据集注册**：

```json
{
  "my_dataset": {
    "file_name": "train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

#### 9.2.2.3 数据划分

将数据划分为训练集、验证集和测试集：

$$
\text{训练集} : \text{验证集} : \text{测试集} = 8 : 1 : 1
$$

**分层采样**：如果数据类别不平衡，按类别分层采样，确保每个集合中各类别比例一致。

```python
from sklearn.model_selection import train_test_split

# 分层划分
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    stratify=[d["label"] for d in data],
    random_state=42
)

val_data, test_data = train_test_split(
    test_data,
    test_size=0.5,
    stratify=[d["label"] for d in test_data],
    random_state=42
)
```

## 9.3 训练配置与调优

### 9.3.1 超参数搜索策略

#### 9.3.1.1 关键超参数

| 超参数      | 含义           | 推荐范围     | 影响                   |
| ----------- | -------------- | ------------ | ---------------------- |
| 学习率      | 参数更新步长   | 1e-5 至 5e-5 | 过大不收敛，过小训练慢 |
| 批次大小    | 每次更新样本数 | 16-128       | 影响梯度估计质量       |
| 训练轮数    | 遍历数据次数   | 1-5          | 过多过拟合             |
| LoRA 秩     | 低秩矩阵秩     | 8-64         | 越大表达能力越强       |
| LoRA alpha  | 缩放因子       | 16-64        | 通常为秩的 2 倍        |
| Warmup 比例 | 预热步数比例   | 0.1          | 稳定训练初期           |
| 权重衰减    | L2 正则化系数  | 0.01-0.1     | 防止过拟合             |
| Dropout     | 随机丢弃概率   | 0.05-0.2     | 防止过拟合             |

#### 9.3.1.2 学习率搜索

**学习率范围测试**：

从很小到很大的学习率分别训练一步，观察损失变化：

```python
lrs = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
losses = []
for lr in lrs:
    model = create_model()
    optimizer = AdamW(model.parameters(), lr=lr)
    loss = train_one_step(model, optimizer, batch)
    losses.append(loss)

# 选择损失下降最快的学习率
best_lr = lrs[np.argmin(losses)]
```

**理论指导**：学习率应满足：

$$
\eta_{\text{opt}} \approx \frac{1}{\sqrt{L}}
$$

其中 $L$ 是损失函数的 Lipschitz 常数。实际中 $L$ 未知，需要通过实验确定。

#### 9.3.1.3 批次大小与学习率的关系

当批次大小增大时，学习率应相应调整。线性缩放规则：

$$
\eta_{\text{new}} = \eta_{\text{base}} \times \frac{B_{\text{new}}}{B_{\text{base}}}
$$

平方根缩放规则：

$$
\eta_{\text{new}} = \eta_{\text{base}} \times \sqrt{\frac{B_{\text{new}}}{B_{\text{base}}}}
$$

实践中，平方根缩放更常用，尤其对于 Adam 优化器。

**推导**：假设梯度估计的方差为 $\sigma^2 / B$，则参数更新的方差为 $\eta^2 \sigma^2 / B$。为了保持更新方差不变，需要：

$$
\frac{\eta^2}{B} = \text{常数} \implies \eta \propto \sqrt{B}
$$

#### 9.3.1.4 超参数搜索方法

**（1）网格搜索（Grid Search）**

枚举所有超参数组合。适用于超参数数量少（≤3）的情况。

```python
from itertools import product

param_grid = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "lora_rank": [8, 16, 32],
    "batch_size": [16, 32],
}

best_params = None
best_score = 0

for params in product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    model = train_model(**config)
    score = evaluate_model(model)
    if score > best_score:
        best_score = score
        best_params = config
```

**（2）随机搜索（Random Search）**

从超参数分布中随机采样。适用于超参数数量多（>3）的情况。

```python
import random

best_params = None
best_score = 0

for _ in range(50):
    config = {
        "learning_rate": 10 ** random.uniform(-6, -4),
        "lora_rank": random.choice([8, 16, 32, 64]),
        "batch_size": random.choice([16, 32, 64]),
        "weight_decay": 10 ** random.uniform(-3, -1),
    }
    model = train_model(**config)
    score = evaluate_model(model)
    if score > best_score:
        best_score = score
        best_params = config
```

**（3）贝叶斯优化**

使用高斯过程建模超参数与性能的关系，选择最有希望的超参数进行评估。

```python
from skopt import gp_minimize

def objective(params):
    learning_rate, lora_rank = params
    model = train_model(learning_rate=learning_rate, lora_rank=lora_rank)
    return -evaluate_model(model)

result = gp_minimize(
    objective,
    dimensions=[
        (1e-6, 1e-4),  # learning_rate
        (4, 64),        # lora_rank
    ],
    n_calls=30,
    random_state=42
)
```

### 9.3.2 训练监控

#### 9.3.2.1 监控指标

| 指标       | 含义           | 正常表现       | 异常表现         |
| ---------- | -------------- | -------------- | ---------------- |
| 训练损失   | 训练集上的损失 | 平稳下降       | 震荡、NaN        |
| 验证损失   | 验证集上的损失 | 先降后升       | 持续上升         |
| 学习率     | 当前学习率     | 按调度变化     | 异常值           |
| 梯度范数   | 梯度的 L2 范数 | 稳定或缓慢变化 | 爆炸或消失       |
| GPU 利用率 | GPU 使用率     | 80%-100%       | 过低（瓶颈）     |
| 显存占用   | 显存使用量     | 稳定           | 逐渐增长（泄漏） |

#### 9.3.2.2 损失曲线分析

**正常训练**：

```
训练损失：持续下降
验证损失：先下降，后趋于平稳
```

**过拟合**：

```
训练损失：持续下降
验证损失：先下降，后上升
→ 解决方案：早停、增加正则化、减少训练轮数
```

**欠拟合**：

```
训练损失：下降缓慢或停滞
验证损失：与训练损失接近，但都较高
→ 解决方案：增加模型容量、增加训练轮数、提高学习率
```

**训练不稳定**：

```
训练损失：剧烈震荡或出现 NaN
→ 解决方案：降低学习率、增加 warmup、梯度裁剪
```

#### 9.3.2.3 早停策略

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop
```

#### 9.3.2.4 梯度裁剪

梯度裁剪防止梯度爆炸。常用的两种方法：

**（1）按范数裁剪**

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left( 1, \frac{\text{max\_norm}}{\|\mathbf{g}\|} \right)
$$

**（2）按值裁剪**

$$
g_i \leftarrow \text{clip}(g_i, -\text{max\_value}, \text{max\_value})
$$

**代码实现**：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 9.3.2.5 训练日志记录

使用 Weights & Biases 或 TensorBoard 记录：

```python
import wandb

wandb.init(project="llm-finetuning", config={
    "learning_rate": 2e-5,
    "lora_rank": 16,
    "batch_size": 32,
})

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    wandb.log({
        "train_loss": loss.item(),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "step": step,
    })
```

## 9.4 模型评估与迭代

### 9.4.1 评估流程

#### 9.4.1.1 分层评估框架

```
第一层：自动指标（快速、低成本）
    ↓ 筛选
第二层：LLM-as-Judge（中等成本）
    ↓ 筛选
第三层：人工评估（高成本、高质量）
```

#### 9.4.1.2 自动评估

**分类任务**：

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

predictions = model.predict(test_data)
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average="weighted")
print(classification_report(test_labels, predictions))
```

**生成任务**：

```python
import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)
```

#### 9.4.1.3 LLM-as-Judge

```python
import openai

def llm_judge(question, answer, reference=None):
    prompt = f"""
    请评估以下回答的质量，从 1-5 分打分（5 分最好）。

    问题：{question}
    回答：{answer}
    {'参考答案：' + reference if reference else ''}

    请输出 JSON 格式：{{"score": <1-5>, "reason": "<理由>"}}
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )
    return response.choices[0].message.content
```

#### 9.4.1.4 人工评估

人工评估的维度：

| 维度       | 说明               | 评分标准 |
| ---------- | ------------------ | -------- |
| 准确性     | 回答是否正确       | 1-5 分   |
| 相关性     | 回答是否与问题相关 | 1-5 分   |
| 流畅性     | 语言是否自然流畅   | 1-5 分   |
| 安全性     | 是否包含有害内容   | 是/否    |
| 风格一致性 | 是否符合期望的风格 | 1-5 分   |

### 9.4.2 迭代优化

#### 9.4.2.1 错误分析

收集模型出错的样本，分析错误模式：

```python
def error_analysis(model, test_data):
    errors = []
    for sample in test_data:
        prediction = model.generate(sample["prompt"])
        if not is_correct(prediction, sample["reference"]):
            errors.append({
                "prompt": sample["prompt"],
                "reference": sample["reference"],
                "prediction": prediction,
                "error_type": classify_error(prediction, sample["reference"])
            })
    return errors
```

#### 9.4.2.2 针对性数据补充

根据错误分析结果，补充针对性训练数据：

```
错误类型统计：
- 格式错误：30% → 补充格式规范的样本
- 知识错误：40% → 补充领域知识样本
- 推理错误：20% → 补充推理链样本
- 其他：10% → 人工审核
```

#### 9.4.2.3 迭代训练

```python
# 第一轮训练
model_v1 = train(base_model, dataset_v1)

# 评估
errors_v1 = error_analysis(model_v1, test_data)

# 补充数据
dataset_v2 = dataset_v1 + generate_targeted_data(errors_v1)

# 第二轮训练
model_v2 = train(base_model, dataset_v2)

# 对比
compare_models(model_v1, model_v2, test_data)
```

#### 9.4.2.4 迭代停止条件

当满足以下条件时停止迭代：

- 评估指标达到目标。
- 连续两轮迭代性能提升 < 1%。
- 人工评估认为质量可接受。
- 成本超过预算。

## 9.5 模型部署与推理

### 9.5.1 LoRA 权重的合并

#### 9.5.1.1 合并的数学原理

LoRA 的前向传播为：

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

其中 $W_0$ 是原始权重，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$\alpha$ 是缩放因子，$r$ 是秩。

合并的目标是将 LoRA 权重融入原始权重，得到新的权重矩阵 $W_{\text{merged}}$，使得：

$$
W_{\text{merged}} x = W_0 x + \frac{\alpha}{r} B A x
$$

因此：

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

合并后，推理时不再需要 LoRA 旁路，计算量与原模型完全一致，无额外延迟。

#### 9.5.1.2 合并的代码实现

使用 PEFT 库合并：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# 加载 LoRA 适配器
peft_model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 合并权重
merged_model = peft_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./merged_model")
```

手动实现合并：

```python
def merge_lora_weights(base_model, lora_state_dict, alpha, r):
    """手动合并 LoRA 权重"""
    for name, param in base_model.named_parameters():
        if name in lora_state_dict:
            lora_A = lora_state_dict[name.replace("weight", "lora_A")]
            lora_B = lora_state_dict[name.replace("weight", "lora_B")]
            param.data += (alpha / r) * (lora_B @ lora_A)
    return base_model
```

#### 9.5.1.3 合并的注意事项

- **精度**：合并时使用 FP32 精度，避免精度损失。
- **不可逆**：合并后无法恢复 LoRA 权重，建议保留原始适配器。
- **多适配器**：如果模型有多个适配器，需要分别合并或动态加载。

### 9.5.2 量化部署

#### 9.5.2.1 量化的数学原理

量化是将浮点权重映射为低精度整数的过程。以 4bit 量化为例：

$$
W \approx s \cdot q
$$

其中 $s$ 是缩放因子（scale），$q$ 是 4bit 整数（范围 0-15 或 -8-7）。

**对称量化**：

$$
s = \frac{\max(|W|)}{2^{b-1} - 1}
$$

$$
q = \text{round}\left( \frac{W}{s} \right)
$$

**非对称量化**：

$$
s = \frac{\max(W) - \min(W)}{2^b - 1}
$$

$$
z = \text{round}\left( \frac{-\min(W)}{s} \right)
$$

$$
q = \text{round}\left( \frac{W}{s} \right) + z
$$

其中 $z$ 是零点（zero point）。

**量化误差**：

$$
\epsilon = W - s \cdot q
$$

量化误差的方差与量化位数的关系：

$$
\text{Var}(\epsilon) \propto \frac{1}{2^{2b}}
$$

即每增加 1bit，量化误差降低为原来的 1/4。

#### 9.5.2.2 GPTQ 量化

GPTQ（Generative Pre-trained Transformer Quantization）是一种针对 LLM 的后训练量化方法。其核心思想是：逐层量化权重，并用校准数据最小化量化误差。

**目标函数**：

$$
\min_{\hat{W}} \| W X - \hat{W} X \|_F^2
$$

其中 $W$ 是原始权重，$\hat{W}$ 是量化后的权重，$X$ 是校准数据。

**GPTQ 的求解**：使用 Hessian 矩阵 $H = 2XX^\top$ 进行迭代更新：

$$
\hat{W}_i = \text{quantize}\left( W_i - \frac{1}{H_{ii}} (W_{i,:} - \hat{W}_{i,:}) H_{:,i} \right)
$$

#### 9.5.2.3 AWQ 量化

AWQ（Activation-aware Weight Quantization）的核心观察是：**不是所有权重都同等重要，应该保护那些对激活值影响大的权重。**

**核心思想**：根据激活值的分布，对重要权重进行缩放，使其在量化后误差更小。

**缩放变换**：

$$
W' = W \cdot \text{diag}(s), \quad X' = X \cdot \text{diag}(s)^{-1}
$$

其中 $s$ 是缩放因子，根据激活值的大小确定。缩放后，$W'X' = WX$，但 $W'$ 的量化误差更小。

#### 9.5.2.4 代码示例

**GPTQ 量化**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# 配置 GPTQ 量化
quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=tokenizer,
    group_size=128,
)

# 加载并量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**AWQ 量化**：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-3-8B"
quant_path = "llama-3-8b-awq"

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 量化配置
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4}

# 量化
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
```

### 9.5.3 推理服务

#### 9.5.3.1 vLLM 部署

vLLM 是目前最流行的高吞吐推理框架，支持 PagedAttention 和连续批处理。

**启动服务**：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000
```

**客户端调用**：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="./merged_model",
    messages=[{"role": "user", "content": "什么是机器学习？"}],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

#### 9.5.3.2 TGI 部署

TGI（Text Generation Inference）是 Hugging Face 的推理框架。

**启动服务**：

```bash
docker run --gpus all -p 8080:80 \
    -v ./merged_model:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /data \
    --max-input-length 2048 \
    --max-total-tokens 4096
```

#### 9.5.3.3 推理性能优化

**（1）批处理**

将多个请求合并为一个批次，提高 GPU 利用率。

**（2）KV Cache**

缓存已计算的键值对，避免重复计算。

**（3）量化推理**

使用 4bit 或 8bit 量化模型，减少显存占用和计算量。

**（4）张量并行**

将模型切分到多个 GPU 上，支持更大模型。

**（5）推测解码**

使用小模型生成草稿，大模型验证，加速生成。

#### 9.5.3.4 吞吐量与延迟的权衡

**吞吐量**：

$$
\text{Throughput} = \frac{\text{Total Tokens}}{\text{Total Time}}
$$

**延迟**：

$$
\text{Latency} = \text{TTFT} + \text{TPOT} \times \text{Output Length}
$$

其中 TTFT 是首 token 延迟，TPOT 是每 token 延迟。

**优化策略**：

| 目标   | 策略                       |
| ------ | -------------------------- |
| 高吞吐 | 大批次、连续批处理、量化   |
| 低延迟 | 小批次、推测解码、KV Cache |
| 平衡   | 动态批处理、自适应调度     |

#### 9.5.3.5 部署检查清单

- [ ] 模型已合并 LoRA 权重（如适用）。
- [ ] 模型已量化（如适用）。
- [ ] 推理框架已配置（vLLM/TGI）。
- [ ] API 服务已启动并测试。
- [ ] 监控已配置（延迟、吞吐、错误率）。
- [ ] 限流和熔断已配置。
- [ ] 日志已配置。
- [ ] 安全措施已配置（认证、授权、内容过滤）。



# 十、典型微调场景实战

## 10.1 领域问答系统

### 10.1.1 场景描述

#### 10.1.1.1 业务背景

领域问答系统是指针对特定专业领域（如医疗、法律、金融、教育等）构建的智能问答应用。用户以自然语言提出问题，系统从领域知识中检索或生成准确答案。

**典型应用**：

- 医疗问答：患者咨询症状、用药、治疗方案
- 法律咨询：合同条款解释、法律条文查询
- 金融顾问：理财产品对比、风险评估
- 教育辅导：学科知识解答、习题讲解

#### 10.1.1.2 核心挑战

**（1）专业术语密集**

领域问答涉及大量专业术语。例如医疗领域的“心肌梗死”、“冠状动脉粥样硬化”，法律领域的“不可抗力”、“缔约过失责任”。通用模型可能无法准确理解和使用这些术语。

**（2）知识准确性要求高**

领域问答的错误代价高。医疗领域的错误建议可能危及生命，法律领域的错误解释可能导致诉讼失败。模型必须基于准确的知识回答问题。

**（3）推理链复杂**

领域问答常需要多步推理。例如医疗诊断需要：症状分析 → 可能疾病 → 鉴别诊断 → 检查建议 → 治疗方案。

**（4）合规与安全**

医疗、法律、金融领域有严格的合规要求。模型必须遵守行业规范，不能给出违规建议。

#### 10.1.1.3 数据特点

| 特点       | 说明                         |
| ---------- | ---------------------------- |
| 专业性强   | 需要领域专家参与标注         |
| 数据量有限 | 高质量领域数据获取成本高     |
| 长尾分布   | 常见问题集中，但长尾问题多样 |
| 多轮对话   | 需要追问和澄清               |
| 引用需求   | 回答需要引用权威来源         |

### 10.1.2 技术方案

#### 10.1.2.1 整体架构

```
用户问题
    ↓
意图识别（判断问题类型）
    ↓
知识检索（从领域知识库检索相关文档）
    ↓
上下文增强（将检索结果注入提示词）
    ↓
LLM 生成（微调后的模型生成回答）
    ↓
引用溯源（标注答案来源）
    ↓
安全过滤（检查合规性）
    ↓
最终回答
```

#### 10.1.2.2 数据准备

**（1）数据来源**

- 领域教科书和指南
- 专家问答记录
- 学术论文摘要
- 临床指南/法律法规/金融报告
- 专家标注的问答对

**（2）数据格式**

```json
[
  {
    "instruction": "患者男性，55岁，突发胸痛2小时，伴大汗、恶心，既往有高血压病史。最可能的诊断是什么？",
    "input": "",
    "output": "根据患者的症状（突发胸痛、大汗、恶心）和病史（高血压），最可能的诊断是急性心肌梗死。建议立即进行心电图检查和心肌酶谱检测，必要时行冠状动脉造影。",
    "source": "《内科学》第9版，第3章",
    "confidence": "high"
  }
]
```

**（3）数据标注要求**

| 要求   | 说明                   |
| ------ | ---------------------- |
| 准确性 | 必须由领域专家审核     |
| 完整性 | 覆盖常见问题和边界情况 |
| 引用性 | 每个回答标注来源       |
| 安全性 | 包含拒答不当问题的样本 |
| 多轮性 | 包含追问和澄清的样本   |

**（4）数据规模建议**

| 子领域数量 | 每个子领域样本数 | 总样本数    |
| ---------- | ---------------- | ----------- |
| 5-10 个    | 500-1000         | 2500-10000  |
| 10-20 个   | 500-1000         | 5000-20000  |
| 20+ 个     | 300-500          | 6000-10000+ |

#### 10.1.2.3 模型选择与微调

**（1）基础模型选择**

| 模型          | 参数量 | 中文能力 | 推荐场景       |
| ------------- | ------ | -------- | -------------- |
| Qwen2.5-7B    | 7B     | 优秀     | 中文领域问答   |
| Qwen2.5-14B   | 14B    | 优秀     | 复杂领域推理   |
| Baichuan2-13B | 13B    | 良好     | 中文对话       |
| LLaMA-3-8B    | 8B     | 一般     | 英文领域问答   |
| Yi-34B        | 34B    | 优秀     | 高精度领域问答 |

**（2）微调方法**

- **方法**：LoRA + SFT（领域知识注入）+ DPO（回答质量对齐）
- **LoRA 配置**：r=32, alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
- **学习率**：1e-5
- **训练轮数**：2-3

**（3）训练损失**

SFT 阶段的损失函数为：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid y_{i,<t}, x_i)
$$

DPO 阶段的损失函数为：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

#### 10.1.2.4 RAG 增强

**（1）知识库构建**

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 加载领域文档
loader = PyPDFLoader("medical_guidelines.pdf")
documents = loader.load()

# 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", ".", " ", ""]
)
chunks = splitter.split_documents(documents)

# 向量化存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./medical_db")
```

**（2）检索与生成**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 提示词模板
prompt = ChatPromptTemplate.from_template("""
你是一个医疗问答助手。请根据以下医学文献回答问题。

医学文献：
{context}

患者问题：{question}

回答要求：
1. 仅基于文献内容回答，不要使用你自身的知识。
2. 如果文献中没有相关信息，请回答"根据现有资料无法回答"。
3. 在回答中标注来源。
4. 对于紧急情况，建议立即就医。

回答：
""")

# 构建 RAG 链
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("高血压患者应该注意什么？")
```

#### 10.1.2.5 评估

| 评估维度   | 指标                   | 目标值 |
| ---------- | ---------------------- | ------ |
| 准确性     | 专家评估准确率         | ≥ 90%  |
| 完整性     | 覆盖关键点的比例       | ≥ 85%  |
| 安全性     | 有害回答比例           | ≤ 1%   |
| 引用准确性 | 引用来源正确率         | ≥ 95%  |
| 拒答率     | 对超出范围问题的拒答率 | ≥ 95%  |

## 10.2 文本分类与信息抽取

### 10.2.1 场景描述

#### 10.2.1.1 业务背景

文本分类是将文本映射到预定义类别的任务，信息抽取是从文本中提取结构化信息的任务。两者常结合使用，例如从新闻中抽取事件类型和参与者。

**典型应用**：

- 情感分析：判断评论是正面还是负面
- 意图识别：判断用户查询的意图类别
- 垃圾邮件检测：判断邮件是否为垃圾邮件
- 命名实体识别：抽取人名、地名、机构名
- 关系抽取：抽取实体之间的关系
- 事件抽取：抽取事件类型、触发词、参与者

#### 10.2.1.2 核心挑战

**（1）类别不平衡**

某些类别的样本远少于其他类别。例如垃圾邮件检测中，垃圾邮件通常只占 5%-10%。

**（2）多标签**

一个文本可能属于多个类别。例如一条新闻可能同时属于“科技”和“金融”。

**（3）细粒度分类**

类别之间的差异可能非常细微。例如“正面”和“中性”的界限模糊。

**（4）结构化输出**

信息抽取要求模型输出结构化数据（如 JSON），对格式准确性要求高。

#### 10.2.1.3 数据特点

| 特点     | 说明                         |
| -------- | ---------------------------- |
| 标注简单 | 相比生成任务，分类标注成本低 |
| 数据量大 | 可以有数万到数百万条样本     |
| 类别固定 | 类别集合预先定义             |
| 评估客观 | 准确率、F1 等指标明确        |

### 10.2.2 技术方案

#### 10.2.2.1 分类任务的技术方案

**（1）数据格式**

```json
[
  {
    "text": "这个电影非常好看，演员表演出色，剧情紧凑。",
    "label": "positive"
  },
  {
    "text": "剧情拖沓，演员演技尴尬，浪费时间。",
    "label": "negative"
  }
]
```

**（2）模型架构**

使用 `AutoModelForSequenceClassification`：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-7B",
    num_labels=3  # positive, negative, neutral
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 数据预处理
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
```

**（3）损失函数**

标准交叉熵损失：

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log \hat{y}_{i,k}
$$

其中 $y_{i,k}$ 是第 $i$ 个样本的第 $k$ 类真实标签（one-hot），$\hat{y}_{i,k}$ 是预测概率。

**（4）类别不平衡处理**

使用加权交叉熵：

$$
\mathcal{L}_{\text{WCE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} w_k \cdot y_{i,k} \log \hat{y}_{i,k}
$$

其中 $w_k$ 是第 $k$ 类的权重，通常取：

$$
w_k = \frac{N}{K \cdot N_k}
$$

$N_k$ 是第 $k$ 类的样本数。推导：权重与类别频率成反比，使得每个类别对损失的贡献大致相等。

**（5）Focal Loss**

对于极不平衡的数据，可以使用 Focal Loss：

$$
\mathcal{L}_{\text{Focal}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \alpha_k (1 - \hat{y}_{i,k})^\gamma y_{i,k} \log \hat{y}_{i,k}
$$

其中 $\gamma$ 是聚焦参数（通常取 2），$\alpha_k$ 是类别权重。

**推导**：Focal Loss 的核心思想是降低易分类样本的权重。对于易分类样本（$\hat{y}_{i,k} \to 1$），$(1 - \hat{y}_{i,k})^\gamma \to 0$，损失被降低。对于难分类样本（$\hat{y}_{i,k} \to 0$），$(1 - \hat{y}_{i,k})^\gamma \to 1$，损失保持较高。

#### 10.2.2.2 信息抽取的技术方案

**（1）命名实体识别**

数据格式：

```json
[
  {
    "text": "张三在北京市朝阳区工作。",
    "entities": [
      {"text": "张三", "type": "PERSON", "start": 0, "end": 2},
      {"text": "北京市", "type": "LOCATION", "start": 3, "end": 6},
      {"text": "朝阳区", "type": "LOCATION", "start": 6, "end": 9}
    ]
  }
]
```

模型架构：使用 `AutoModelForTokenClassification`：

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "Qwen/Qwen2.5-7B",
    num_labels=len(label_list)  # BIO 标注的标签数
)
```

**（2）关系抽取**

数据格式：

```json
[
  {
    "text": "张三是北京大学的教授。",
    "relations": [
      {"head": "张三", "tail": "北京大学", "relation": "works_at"}
    ]
  }
]
```

**（3）结构化输出**

使用提示词让模型生成 JSON：

```python
prompt = """
请从以下文本中抽取实体和关系，以 JSON 格式输出。

文本：{text}

输出格式：
{
  "entities": [{"text": "...", "type": "..."}],
  "relations": [{"head": "...", "tail": "...", "relation": "..."}]
}
"""
```

使用 Pydantic 定义输出结构：

```python
from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    text: str = Field(description="实体文本")
    type: str = Field(description="实体类型")

class Relation(BaseModel):
    head: str = Field(description="头实体")
    tail: str = Field(description="尾实体")
    relation: str = Field(description="关系类型")

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(description="实体列表")
    relations: List[Relation] = Field(description="关系列表")

structured_model = model.with_structured_output(ExtractionResult)
result = structured_model.invoke(f"请从以下文本中抽取信息：{text}")
```

#### 10.2.2.3 评估

| 任务       | 指标                 | 说明                             |
| ---------- | -------------------- | -------------------------------- |
| 分类       | 准确率、F1、AUC      | F1 对不平衡数据更敏感            |
| NER        | 实体级 F1            | 精确匹配实体边界和类型           |
| 关系抽取   | 关系级 F1            | 精确匹配头实体、尾实体和关系类型 |
| 结构化输出 | 格式正确率 + 内容 F1 | 格式正确率和内容准确率           |

## 10.3 对话风格定制

### 10.3.1 场景描述

#### 10.3.1.1 业务背景

对话风格定制是指让模型的输出符合特定的品牌调性、角色设定或交流风格。例如，客服机器人需要友好、专业；品牌代言需要符合品牌个性；游戏 NPC 需要符合角色设定。

**典型应用**：

- 品牌客服：符合品牌调性的专业回复
- 角色扮演：模仿特定角色的语言风格
- 情感陪伴：温暖、共情的对话风格
- 教育辅导：耐心、鼓励的教学风格
- 幽默助手：轻松、幽默的交流方式

#### 10.3.1.2 核心挑战

**（1）风格难以量化**

风格是主观的、多维度的，难以用简单的规则描述。

**（2）风格与内容分离**

模型需要在不改变内容准确性的前提下改变风格。

**（3）一致性**

模型需要在多轮对话中保持风格一致。

**（4）个性化**

不同用户可能期望不同的风格。

#### 10.3.1.3 数据特点

| 特点     | 说明                     |
| -------- | ------------------------ |
| 风格标注 | 需要标注者判断风格一致性 |
| 对比数据 | 同一内容的不同风格版本   |
| 多轮对话 | 风格在对话中体现         |
| 主观性强 | 风格评价因人而异         |

### 10.3.2 技术方案

#### 10.3.2.1 数据准备

**（1）风格示范数据**

```json
[
  {
    "instruction": "用户询问退货政策",
    "output": "亲，感谢您的咨询！我们的退货政策非常人性化哦～收到商品后7天内，只要不影响二次销售，都可以无理由退货呢。有任何问题随时找我，我会全程帮您处理的！😊",
    "style": "friendly_casual"
  },
  {
    "instruction": "用户询问退货政策",
    "output": "您好，根据我司退货政策，商品签收后7日内，在不影响二次销售的前提下，可申请无理由退货。如需办理，请提供订单编号，我们将尽快为您处理。",
    "style": "formal_professional"
  }
]
```

**（2）偏好对比数据（用于 DPO）**

```json
[
  {
    "prompt": "用户询问退货政策",
    "chosen": "亲，感谢您的咨询！我们的退货政策非常人性化哦～收到商品后7天内，只要不影响二次销售，都可以无理由退货呢。",
    "rejected": "退货政策：7天内可退。",
    "style": "friendly_casual"
  }
]
```

#### 10.3.2.2 微调方法

**（1）SFT 阶段**

使用风格示范数据进行监督微调：

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./style_sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=style_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

**（2）DPO 阶段**

使用偏好对比数据进一步对齐风格：

```python
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./style_dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    beta=0.1,
    bf16=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

#### 10.3.2.3 风格控制的数学建模

**（1）风格作为条件概率**

将风格 $s$ 作为条件，模型生成的概率为：

$$
P(y \mid x, s) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x, s)
$$

其中 $s$ 是风格标签（如 "friendly_casual"）。

**（2）风格嵌入**

将风格标签映射为嵌入向量：

$$
\mathbf{e}_s = \text{Embedding}(s) \in \mathbb{R}^d
$$

将风格嵌入拼接到输入嵌入中：

$$
\mathbf{h}_0 = [\mathbf{e}_x; \mathbf{e}_s]
$$

**（3）风格强度控制**

引入风格强度参数 $\lambda$，控制风格的强弱：

$$
P(y \mid x, s, \lambda) \propto P(y \mid x, s)^\lambda \cdot P(y \mid x)^{1-\lambda}
$$

当 $\lambda = 1$ 时，完全使用风格化模型；当 $\lambda = 0$ 时，使用通用模型。

**推导**：对两边取对数：

$$
\log P(y \mid x, s, \lambda) = \lambda \log P(y \mid x, s) + (1-\lambda) \log P(y \mid x) + C
$$

这相当于在 logits 层面进行线性插值：

$$
z_{\text{combined}} = \lambda z_{\text{style}} + (1-\lambda) z_{\text{general}}
$$

#### 10.3.2.4 评估

| 评估维度   | 方法         | 说明                     |
| ---------- | ------------ | ------------------------ |
| 风格一致性 | LLM-as-Judge | 判断输出是否符合目标风格 |
| 内容准确性 | 人工评估     | 判断内容是否正确         |
| 风格强度   | 人工评分     | 1-5 分评估风格强度       |
| 多轮一致性 | 对话级评估   | 评估整个对话的风格一致性 |

## 10.4 推理能力增强

### 10.4.1 场景描述

#### 10.4.1.1 业务背景

推理能力增强是指提升模型在数学、逻辑、代码等需要多步推理的任务上的表现。这类任务要求模型不仅给出答案，还要展示推理过程。

**典型应用**：

- 数学解题：从小学算术到高等数学
- 代码生成：根据需求生成可运行的代码
- 逻辑推理：解决逻辑谜题、推理题
- 科学推理：物理、化学、生物问题
- 数据分析：从数据中得出结论

#### 10.4.1.2 核心挑战

**（1）多步推理**

复杂问题需要多步推理，每一步都可能出错。

**（2）验证困难**

推理过程的正确性难以自动验证，尤其是开放式问题。

**（3）数据获取**

高质量的推理链数据获取成本高。

**（4）泛化能力**

模型需要泛化到未见过的题型。

#### 10.4.1.3 数据特点

| 特点     | 说明                   |
| -------- | ---------------------- |
| 推理链   | 需要包含完整的推理步骤 |
| 可验证   | 最终答案可以程序化验证 |
| 多样性   | 需要覆盖多种题型       |
| 难度分级 | 从简单到复杂           |

### 10.4.2 技术方案

#### 10.4.2.1 数据准备

**（1）推理链数据格式**

```json
[
  {
    "instruction": "一个水池有两个进水管和一个出水管。甲管单独注水需要6小时，乙管单独注水需要8小时，丙管单独排水需要12小时。三管同时打开，需要多少小时注满水池？",
    "output": "设水池容量为1。\n\n甲管每小时注水：1/6\n乙管每小时注水：1/8\n丙管每小时排水：1/12\n\n三管同时打开，每小时净注水：\n1/6 + 1/8 - 1/12\n\n通分（分母24）：\n4/24 + 3/24 - 2/24 = 5/24\n\n注满水池需要：\n1 ÷ 5/24 = 24/5 = 4.8小时\n\n答案：4.8小时",
    "answer": "4.8",
    "type": "math"
  }
]
```

**（2）推理链生成**

使用强 LLM 生成推理链，再人工审核：

```python
def generate_reasoning_chain(problem):
    prompt = f"""
    请逐步解答以下问题，展示完整的推理过程。

    问题：{problem}

    要求：
    1. 每一步都要有清晰的解释
    2. 使用数学公式时要用 LaTeX 格式
    3. 最后给出明确的答案

    解答：
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content
```

#### 10.4.2.2 SFT 阶段

使用推理链数据进行监督微调：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{|y_i|} \log P_\theta(y_{i,t} \mid y_{i,<t}, x_i)
$$

其中 $y_i$ 是完整的推理链（包括推理步骤和最终答案）。

#### 10.4.2.3 GRPO 阶段

使用 GRPO 结合可验证奖励进一步提升推理能力：

**（1）奖励函数设计**

```python
def math_reward(prompt, completion, ground_truth):
    """数学题的奖励函数"""
    import re
    # 提取最终答案
    answer_match = re.search(r"答案[：:]\s*([\d\.]+)", completion)
    if not answer_match:
        return 0.0
    predicted = float(answer_match.group(1))
    # 比较答案
    if abs(predicted - float(ground_truth)) < 1e-6:
        return 1.0
    else:
        return 0.0

def format_reward(completion):
    """格式奖励：检查推理链是否完整"""
    if "步骤" in completion and "答案" in completion:
        return 0.5
    return 0.0
```

**（2）GRPO 训练**

```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    output_dir="./grpo_math",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_generations=8,  # 每个问题采样8个输出
    beta=0.04,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=math_dataset,
    reward_funcs=[math_reward, format_reward],
    tokenizer=tokenizer,
)
trainer.train()
```

#### 10.4.2.4 推理链的数学建模

**（1）思维链（Chain-of-Thought, CoT）**

将推理过程分解为一系列中间步骤 $z_1, z_2, \dots, z_K$，最终答案 $y$ 由这些步骤推导得出：

$$
P(y \mid x) = \sum_{z_1, \dots, z_K} P(z_1 \mid x) P(z_2 \mid x, z_1) \cdots P(y \mid x, z_1, \dots, z_K)
$$

**（2）自一致性（Self-Consistency）**

采样多条推理链，取多数答案：

$$
\hat{y} = \arg\max_y \sum_{k=1}^{K} \mathbb{1}[y_k = y]
$$

其中 $y_k$ 是第 $k$ 条推理链的答案。

**推导**：自一致性基于以下假设：正确的推理链更可能得出一致的答案。通过多数投票，可以过滤掉偶然的错误。

**（3）过程奖励模型（Process Reward Model, PRM）**

对推理链的每一步进行评分：

$$
r_{\text{PRM}}(z_t \mid x, z_{<t}) = \text{Score of step } t
$$

最终奖励为各步奖励的乘积或最小值：

$$
r_{\text{total}} = \prod_{t=1}^{K} r_{\text{PRM}}(z_t \mid x, z_{<t})
$$

或：

$$
r_{\text{total}} = \min_{t} r_{\text{PRM}}(z_t \mid x, z_{<t})
$$

#### 10.4.2.5 评估

| 评估维度   | 指标           | 说明             |
| ---------- | -------------- | ---------------- |
| 答案准确性 | 准确率         | 最终答案是否正确 |
| 推理正确性 | 步骤正确率     | 推理步骤是否正确 |
| 泛化能力   | 未见题型准确率 | 在新题型上的表现 |
| 效率       | 平均推理步数   | 推理链的长度     |

## 10.5 多模态微调

### 10.5.1 场景描述

#### 10.5.1.1 业务背景

多模态微调是指对能够处理多种模态（文本、图像、音频、视频）的大模型进行微调，使其在特定任务上表现更好。

**典型应用**：

- 视觉问答：根据图片回答问题
- 图像描述：为图片生成文字描述
- 文档理解：从扫描文档中提取信息
- 图表分析：理解图表并回答相关问题
- 视频理解：分析视频内容并生成摘要

#### 10.5.1.2 核心挑战

**（1）模态对齐**

不同模态的信息需要对齐。例如，图像中的物体需要与文本中的词汇对应。

**（2）数据稀缺**

多模态数据（如图文对）的标注成本高。

**（3）计算资源**

多模态模型通常参数量大，训练和推理成本高。

**（4）评估困难**

多模态任务的评估比纯文本任务更复杂。

#### 10.5.1.3 数据特点

| 特点     | 说明                        |
| -------- | --------------------------- |
| 多模态   | 包含图像、文本等多种模态    |
| 对齐标注 | 需要标注图像-文本的对应关系 |
| 任务多样 | VQA、描述生成、文档理解等   |
| 规模大   | 图像数据通常较大            |

### 10.5.2 技术方案

#### 10.5.2.1 模型架构

**（1）典型多模态架构**

```
图像 → 视觉编码器（ViT）→ 视觉特征
                                    ↓
文本 → 文本编码器 → 文本特征 → 融合层 → LLM → 输出
```

**（2）常用多模态模型**

| 模型      | 视觉编码器 | LLM           | 参数量 |
| --------- | ---------- | ------------- | ------ |
| LLaVA-1.5 | CLIP ViT-L | Vicuna-7B/13B | 7B/13B |
| Qwen-VL   | ViT        | Qwen-7B       | 7B     |
| InternVL  | InternViT  | InternLM      | 7B-20B |
| CogVLM    | EVA-CLIP   | Vicuna        | 17B    |

#### 10.5.2.2 数据准备

**（1）视觉问答数据**

```json
[
  {
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\n这张图片中有什么？"},
      {"from": "gpt", "value": "这张图片展示了一只橘猫坐在窗台上，窗外是城市的天际线。"}
    ]
  }
]
```

**（2）文档理解数据**

```json
[
  {
    "image": "path/to/document.png",
    "conversations": [
      {"from": "human", "value": "<image>\n请提取这张发票的金额和日期。"},
      {"from": "gpt", "value": "金额：￥1,250.00\n日期：2026-01-15"}
    ]
  }
]
```

#### 10.5.2.3 微调方法

**（1）LoRA 微调**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 加载多模态模型
model = AutoModelForCausalLM.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, lora_config)
```

**（2）训练**

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./multimodal_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=multimodal_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

#### 10.5.2.4 MDPO 多模态偏好优化

使用 MDPO 进一步对齐多模态输出：

$$
\mathcal{L}_{\text{MDPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \sum_{k=1}^{K} w_k \left( \hat{r}_k(x, y_w) - \hat{r}_k(x, y_l) \right) \right) \right]
$$

其中 $\hat{r}_k(x, y) = \beta_k \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$，$K$ 个维度包括：

- 文本准确性
- 图像-文本对齐度
- 语言流畅性
- 安全性

#### 10.5.2.5 评估

| 评估维度 | 指标                   | 说明                       |
| -------- | ---------------------- | -------------------------- |
| 文本质量 | BLEU、ROUGE、BERTScore | 生成文本的质量             |
| 视觉理解 | VQA 准确率             | 视觉问答的准确率           |
| 对齐度   | 图文一致性评分         | 文本与图像的一致性         |
| 幻觉率   | 幻觉检测               | 生成内容中与图像不符的比例 |
| 安全性   | 有害内容比例           | 生成内容中不安全的比例     |

#### 10.5.2.6 多模态微调的注意事项

| 注意事项       | 说明                                |
| -------------- | ----------------------------------- |
| 视觉编码器冻结 | 通常冻结视觉编码器，只微调 LLM 部分 |
| 图像分辨率     | 高分辨率图像需要更多的计算资源      |
| 批次大小       | 多模态数据的批次大小通常较小        |
| 显存优化       | 使用梯度检查点、混合精度等技术      |
| 数据格式       | 确保图像路径和对话格式正确          |
| 评估多样性     | 多模态任务需要多维度评估            |



# 十一、微调学习资源

## 11.1 核心论文

### 11.1.1 LoRA: Low-Rank Adaptation of Large Language Models

#### 11.1.1.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | LoRA: Low-Rank Adaptation of Large Language Models           |
| 作者 | Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen |
| 发表 | ICLR 2022                                                    |
| 链接 | https://arxiv.org/abs/2106.09685                             |
| 代码 | https://github.com/microsoft/LoRA                            |

#### 11.1.1.2 核心贡献

LoRA 的核心贡献是提出了低秩适应方法，将权重更新分解为两个低秩矩阵的乘积。论文证明了：

1. 微调过程中，权重更新的秩远小于原始权重矩阵的秩。
2. 将更新约束在低秩子空间，可以在保持性能的同时大幅减少可训练参数量。
3. LoRA 权重可以在推理时合并回原模型，无额外延迟。

#### 11.1.1.3 关键公式推导

**（1）低秩假设的数学依据**

论文通过实验观察到，微调后的权重更新 $\Delta W$ 的奇异值分布呈现快速衰减。设 $\Delta W$ 的奇异值分解为：

$$
\Delta W = \sum_{i=1}^{\min(d,k)} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

其中 $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$ 是奇异值。实验发现，前 $r$ 个奇异值（$r \ll \min(d,k)$）已经捕获了 $\Delta W$ 的绝大部分能量：

$$
\frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{\min(d,k)} \sigma_i^2} \geq 1 - \epsilon
$$

其中 $\epsilon$ 是较小的常数。这说明 $\Delta W$ 可以用秩为 $r$ 的矩阵很好地近似：

$$
\Delta W \approx \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = B A
$$

其中 $B = [\sigma_1 \mathbf{u}_1, \dots, \sigma_r \mathbf{u}_r] \in \mathbb{R}^{d \times r}$，$A = [\mathbf{v}_1, \dots, \mathbf{v}_r]^\top \in \mathbb{R}^{r \times k}$。

**（2）前向传播的完整推导**

原始线性层的输出为：

$$
h = W_0 x
$$

加入 LoRA 后，输出变为：

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

**为什么使用 $\alpha/r$ 缩放？**

论文的推导如下：假设 $A$ 的每个元素从 $\mathcal{N}(0, \sigma_A^2)$ 采样，$B$ 初始化为零矩阵。训练过程中，$B$ 的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot \left( \frac{\alpha}{r} \right) \cdot (A x)^\top
$$

$B$ 的更新量与 $\alpha/r$ 成正比。如果 $\alpha$ 固定，$r$ 越大，每次更新的有效幅度越小。为了在不同 $r$ 下保持训练动态一致，需要将 $\alpha/r$ 作为有效缩放因子。

**（3）参数量对比**

| 方法     | 参数量       | 占比（d=k=4096, r=16） |
| -------- | ------------ | ---------------------- |
| 全参微调 | $d \times k$ | 100%                   |
| LoRA     | $r(d+k)$     | 0.78%                  |

**（4）推理时的合并**

合并后的权重为：

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

合并后，推理时的计算为：

$$
h = W_{\text{merged}} x
$$

与原模型的计算完全一致，无额外延迟。

#### 11.1.1.4 论文的实验结论

| 实验                | 结论                                                         |
| ------------------- | ------------------------------------------------------------ |
| GPT-2 上的实验      | LoRA 在参数量减少 10000 倍的情况下，性能与全参微调相当       |
| GPT-3 175B 上的实验 | LoRA 在参数量减少 10000 倍的情况下，性能与全参微调相当或更好 |
| 推理延迟            | 合并后无额外延迟                                             |
| 多任务切换          | 可以快速切换不同任务的 LoRA 适配器                           |

### 11.1.2 QLoRA: Efficient Finetuning of Quantized LLMs

#### 11.1.2.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | QLoRA: Efficient Finetuning of Quantized LLMs                |
| 作者 | Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer |
| 发表 | NeurIPS 2023                                                 |
| 链接 | https://arxiv.org/abs/2305.14314                             |
| 代码 | https://github.com/artidoro/qlora                            |

#### 11.1.2.2 核心贡献

QLoRA 的核心贡献是：

1. 提出了 NF4（NormalFloat 4-bit）量化格式，信息论最优。
2. 提出了双重量化（Double Quantization），进一步减少显存。
3. 提出了分页优化器（Paged Optimizers），避免 OOM。
4. 证明了在单张 48GB GPU 上微调 65B 模型可以达到与全参微调相当的性能。

#### 11.1.2.3 关键公式推导

**（1）NF4 量化的信息论推导**

NF4 的设计基于以下观察：预训练模型的权重近似服从正态分布。对于正态分布的随机变量，最优的量化级别应该按照分位数划分。

设权重 $W \sim \mathcal{N}(0, \sigma^2)$，量化级别为 $q_1 < q_2 < \dots < q_{2^b}$。量化误差的期望为：

$$
\mathbb{E}[\epsilon^2] = \sum_{i=1}^{2^b} \int_{q_i}^{q_{i+1}} (w - c_i)^2 p(w) dw
$$

其中 $c_i$ 是第 $i$ 个量化级别的代表值，$p(w)$ 是权重的概率密度函数。

最小化量化误差，可以得到最优的量化级别应该满足：

$$
c_i = \mathbb{E}[W \mid W \in [q_i, q_{i+1}]]
$$

即每个量化级别的代表值应该是对应区间内权重的条件期望。对于标准正态分布，这等价于：

$$
c_i = \Phi^{-1}\left( \frac{i + 0.5}{2^b} \right)
$$

其中 $\Phi^{-1}$ 是标准正态分布的逆累积分布函数。

**（2）双重量化的推导**

标准量化中，每个块需要存储一个 FP32 的缩放因子。设块大小为 $B$，则每个参数额外需要 $32/B$ bits 的存储。

双重量化对这些缩放因子再进行一次量化。设外层量化使用 $b_2$ bits，块大小为 $B_2$，则每个参数额外需要：

$$
\frac{32}{B} + \frac{b_2}{B \cdot B_2}
$$

bits 的存储。

**推导**：假设有 $n$ 个参数，分成 $n/B$ 个块，每个块一个 FP32 缩放因子，需要 $32n/B$ bits。双重量化将这 $n/B$ 个缩放因子分成 $(n/B)/B_2$ 组，每组一个 FP32 外层缩放因子，需要 $32n/(B \cdot B_2)$ bits，加上内层量化需要 $b_2 \cdot n/(B \cdot B_2)$ bits。总计：

$$
\frac{32n}{B} \to \frac{32n}{B \cdot B_2} + \frac{b_2 \cdot n}{B \cdot B_2}
$$

节省的存储为：

$$
\frac{32n}{B} - \frac{32n}{B \cdot B_2} - \frac{b_2 \cdot n}{B \cdot B_2} = \frac{n}{B} \left( 32 - \frac{32 + b_2}{B_2} \right)
$$

当 $B=64$，$B_2=256$，$b_2=8$ 时，每个参数节省约 0.37 bits。

**（3）QLoRA 的前向传播**

$$
Y^{\mathrm{BF16}} = X^{\mathrm{BF16}} \cdot \mathrm{doubleDequant}(c_1^{\mathrm{FP32}}, c_2^{\mathrm{k-bit}}, W^{\mathrm{NF4}}) + X^{\mathrm{BF16}} L_1^{\mathrm{BF16}} L_2^{\mathrm{BF16}}
$$

其中：

- $\mathrm{doubleDequant}$ 是双重量化的反量化函数
- $c_1$ 是外层 FP32 缩放因子
- $c_2$ 是内层 k-bit 缩放因子
- $W^{\mathrm{NF4}}$ 是 NF4 量化后的权重
- $L_1, L_2$ 是 LoRA 的 $B$ 和 $A$ 矩阵

#### 11.1.2.4 论文的实验结论

| 实验         | 结论                                       |
| ------------ | ------------------------------------------ |
| 65B 模型     | 单张 48GB GPU 可微调，性能与全参微调相当   |
| Guanaco 模型 | 在 Vicuna 基准上达到 99.3% 的 ChatGPT 性能 |
| 训练时间     | 24 小时内完成 65B 模型的微调               |

### 11.1.3 DPO: Direct Preference Optimization

#### 11.1.3.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| 作者 | Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn |
| 发表 | NeurIPS 2023                                                 |
| 链接 | https://arxiv.org/abs/2305.18290                             |

#### 11.1.3.2 核心贡献

DPO 的核心贡献是：

1. 证明了 RLHF 的约束优化目标存在闭式最优解。
2. 将该闭式解转化为简单的分类损失，跳过奖励模型和强化学习。
3. 在多个任务上达到或超越 RLHF 的性能。

#### 11.1.3.3 关键公式推导

**（1）最优策略的闭式解**

RLHF 的优化目标为：

$$
\max_{\pi} \ \mathbb{E}_{y \sim \pi(y \mid x)} \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

约束条件为 $\sum_y \pi(y \mid x) = 1$。使用拉格朗日乘子法，构造拉格朗日函数：

$$
\mathcal{L}(\pi, \lambda) = \sum_y \pi(y \mid x) \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] + \lambda \left( 1 - \sum_y \pi(y \mid x) \right)
$$

对 $\pi(y \mid x)$ 求偏导并令为 0：

$$
r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \beta - \lambda = 0
$$

整理得：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ 是归一化常数。

**（2）奖励函数的反解**

对最优策略取对数：

$$
\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)
$$

整理得：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**（3）Bradley-Terry 模型的代入**

人类偏好概率为：

$$
P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

代入奖励表达式：

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

注意 $\beta \log Z(x)$ 项在相减时抵消。

**（4）DPO 损失**

用 $\pi_\theta$ 替代 $\pi^*$，最大化偏好似然：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

**（5）梯度分析**

对 DPO 损失求梯度：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E} \left[ \sigma\left( \hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w) \right) \left( \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right) \right]
$$

其中 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ 是隐式奖励。

#### 11.1.3.4 论文的实验结论

| 实验       | 结论                           |
| ---------- | ------------------------------ |
| 情感控制   | DPO 在 IMDb 评论生成上优于 PPO |
| 摘要生成   | DPO 在 Reddit TL;DR 上优于 PPO |
| 单轮对话   | DPO 在 Anthropic HH 上优于 PPO |
| 训练稳定性 | DPO 比 PPO 更稳定，超参数更少  |

### 11.1.4 DeepSeekMath (GRPO)

#### 11.1.4.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 作者 | DeepSeek-AI                                                  |
| 发表 | 2024                                                         |
| 链接 | https://arxiv.org/abs/2402.03300                             |

#### 11.1.4.2 核心贡献

GRPO（Group Relative Policy Optimization）的核心贡献是：

1. 移除了 PPO 中的 Critic 网络，使用组内奖励统计估计优势。
2. 大幅降低了显存需求和计算复杂度。
3. 在数学推理任务上取得了显著效果。

#### 11.1.4.3 关键公式推导

**（1）优势函数的组内估计**

在 PPO 中，优势函数为：

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

其中 $V(s_t)$ 由 Critic 网络估计。

GRPO 使用组内平均奖励作为基线：

$$
V(q) \approx \text{mean}(\{r_1, r_2, \dots, r_G\}) = \frac{1}{G} \sum_{i=1}^G r_i
$$

因此，每个输出的优势为：

$$
A_i = r_i - \text{mean}(\{r_1, \dots, r_G\})
$$

标准化后：

$$
A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}
$$

**（2）为什么组内平均是好的基线？**

**推导**：设真实状态价值为 $V^*(q)$。组内平均奖励是 $V^*(q)$ 的无偏估计：

$$
\mathbb{E}[\text{mean}(\{r_1, \dots, r_G\})] = \mathbb{E}[r] = V^*(q)
$$

其方差为：

$$
\text{Var}(\text{mean}) = \frac{\text{Var}(r)}{G}
$$

当 $G$ 增大时，方差减小。因此，组内平均是 $V^*(q)$ 的合理估计，尤其在 $G$ 较大时。

**（3）GRPO 的优化目标**

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i(\theta) A_i, \ \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}}) \right]
$$

其中 $\rho_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}$。

**（4）与 PPO 的显存对比**

| 组件        | PPO  | GRPO |
| ----------- | ---- | ---- |
| 策略模型    | ✓    | ✓    |
| 参考模型    | ✓    | ✓    |
| 奖励模型    | ✓    | ✗    |
| Critic 网络 | ✓    | ✗    |
| 总模型数    | 4    | 2    |
| 显存需求    | 高   | 低   |

#### 11.1.4.4 论文的实验结论

| 实验       | 结论                          |
| ---------- | ----------------------------- |
| GSM8K      | GRPO 显著提升数学推理准确率   |
| MATH       | GRPO 在竞赛数学上取得显著提升 |
| 训练稳定性 | GRPO 比 PPO 更稳定            |
| 显存需求   | GRPO 比 PPO 节省约 50% 显存   |

### 11.1.5 DoRA: Weight-Decomposed Low-Rank Adaptation

#### 11.1.5.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | DoRA: Weight-Decomposed Low-Rank Adaptation                  |
| 作者 | Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen |
| 发表 | ICML 2024                                                    |
| 链接 | https://arxiv.org/abs/2402.09353                             |

#### 11.1.5.2 核心贡献

DoRA 的核心贡献是：

1. 将权重分解为幅度和方向两个分量。
2. 仅对方向分量应用 LoRA。
3. 在低秩设置下优于 LoRA，且无额外推理延迟。

#### 11.1.5.3 关键公式推导

**（1）权重分解**

将权重矩阵 $W \in \mathbb{R}^{d \times k}$ 分解为：

$$
W = m \frac{V}{\|V\|_c}
$$

其中 $m \in \mathbb{R}^{1 \times k}$ 是幅度向量（每列的 L2 范数），$V \in \mathbb{R}^{d \times k}$ 是方向矩阵，$\|V\|_c$ 表示逐列的 L2 范数。

**推导**：对于每一列 $j$，$W_{:,j} = m_j \frac{V_{:,j}}{\|V_{:,j}\|_2}$。其中 $m_j = \|W_{:,j}\|_2$ 是列的幅度，$\frac{V_{:,j}}{\|V_{:,j}\|_2}$ 是单位方向向量。

**（2）DoRA 的参数化**

DoRA 的更新为：

$$
W' = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

其中 $W_0$ 冻结，$B$ 和 $A$ 可训练，$m$ 可训练。

**（3）与 LoRA 的对比**

LoRA：$W_{\text{LoRA}} = W_0 + BA$

DoRA：$W_{\text{DoRA}} = m \frac{W_0 + BA}{\|W_0 + BA\|_c}$

DoRA 增加了幅度 $m$ 的独立学习能力，使模型可以分别调整权重的幅度和方向。

**（4）参数量对比**

LoRA 参数量：$r(d+k)$

DoRA 参数量：$r(d+k) + k$

由于 $k \ll r(d+k)$，额外参数量可忽略。

#### 11.1.5.4 论文的实验结论

| 实验         | 结论                             |
| ------------ | -------------------------------- |
| 常识推理     | DoRA 在 LLaMA-7B/13B 上优于 LoRA |
| 视觉指令调优 | DoRA 在 LLaVA 上优于 LoRA        |
| 低秩设置     | DoRA 在 r=4/8 时优势最明显       |
| 推理延迟     | 无额外延迟                       |

### 11.1.6 InstructGPT (RLHF)

#### 11.1.6.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | Training language models to follow instructions with human feedback |
| 作者 | Long Ouyang, Jeff Wu, Xu Jiang, et al.                       |
| 发表 | NeurIPS 2022                                                 |
| 链接 | https://arxiv.org/abs/2203.02155                             |

#### 11.1.6.2 核心贡献

InstructGPT 的核心贡献是：

1. 提出了 RLHF 的三阶段流程：SFT → RM → PPO。
2. 证明了 RLHF 可以显著提升模型遵循指令的能力。
3. 揭示了奖励黑客现象和 KL 惩罚的重要性。

#### 11.1.6.3 关键公式推导

**（1）奖励模型损失**

Bradley-Terry 模型：

$$
P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

奖励模型损失：

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

**（2）PPO 优化目标**

$$
\max_{\pi_\theta} \ \mathbb{E}_{x, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

**（3）KL 惩罚的作用**

KL 惩罚防止策略偏离参考模型太远。$\beta$ 越大，策略越保守；$\beta$ 越小，策略越激进。

#### 11.1.6.4 论文的实验结论

| 实验     | 结论                             |
| -------- | -------------------------------- |
| 指令遵循 | InstructGPT 显著优于 GPT-3       |
| 真实性   | InstructGPT 的幻觉率更低         |
| 安全性   | InstructGPT 的有害输出更少       |
| 人类偏好 | 1.3B InstructGPT 优于 175B GPT-3 |

## 11.2 官方文档与教程

### 11.2.1 Hugging Face PEFT 文档

#### 11.2.1.1 基本信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 链接 | https://huggingface.co/docs/peft                             |
| 内容 | PEFT 方法的完整文档，包括 LoRA、Prefix Tuning、Prompt Tuning、IA³ 等 |
| 特点 | 代码示例丰富，API 文档详细                                   |

#### 11.2.1.2 核心内容

| 章节     | 内容                                   |
| -------- | -------------------------------------- |
| 快速入门 | 安装、基本用法、第一个 LoRA 微调       |
| 概念指南 | PEFT 方法分类、适用场景                |
| 任务指南 | 分类、生成、序列标注等任务的 PEFT 用法 |
| API 文档 | 所有类和方法详细说明                   |
| 示例     | 完整的端到端示例                       |

#### 11.2.1.3 推荐学习路径

```
1. 阅读"快速入门"章节，完成第一个 LoRA 微调
2. 阅读"概念指南"，理解不同 PEFT 方法的原理
3. 根据自己的任务，阅读对应的"任务指南"
4. 参考 API 文档，深入了解各参数
5. 运行官方示例，加深理解
```

### 11.2.2 Hugging Face TRL 文档

#### 11.2.2.1 基本信息

| 项目 | 内容                                   |
| ---- | -------------------------------------- |
| 链接 | https://huggingface.co/docs/trl        |
| 内容 | SFT、DPO、GRPO、PPO 等训练器的完整文档 |
| 特点 | 与 Transformers、PEFT 无缝集成         |

#### 11.2.2.2 核心内容

| 章节           | 内容                     |
| -------------- | ------------------------ |
| SFT Trainer    | 监督微调的完整指南       |
| DPO Trainer    | 直接偏好优化的完整指南   |
| GRPO Trainer   | 组相对策略优化的完整指南 |
| PPO Trainer    | 近端策略优化的完整指南   |
| Reward Trainer | 奖励模型训练的完整指南   |

#### 11.2.2.3 推荐学习路径

```
1. 阅读 SFT Trainer 文档，完成监督微调
2. 阅读 DPO Trainer 文档，完成偏好优化
3. 阅读 GRPO Trainer 文档，了解无 Critic 的 RL
4. 参考 API 文档，深入了解各参数
5. 运行官方示例，加深理解
```

### 11.2.3 LLaMA-Factory GitHub

#### 11.2.3.1 基本信息

| 项目 | 内容                                         |
| ---- | -------------------------------------------- |
| 链接 | https://github.com/hiyouga/LLaMA-Factory     |
| 内容 | 集成化微调框架，支持 100+ 模型和多种微调方法 |
| 特点 | Web UI、命令行接口、丰富的数据集支持         |

#### 11.2.3.2 核心内容

| 模块   | 内容                            |
| ------ | ------------------------------- |
| Web UI | 可视化配置和训练                |
| 命令行 | 脚本化训练                      |
| 数据集 | 100+ 数据集支持                 |
| 模型   | 100+ 模型支持                   |
| 方法   | 全参、LoRA、QLoRA、DPO、GRPO 等 |

#### 11.2.3.3 推荐学习路径

```
1. 阅读 README，了解项目结构和功能
2. 按照 Quick Start 完成第一个微调
3. 学习 Web UI 的使用
4. 学习命令行参数配置
5. 参考高级教程，进行自定义开发
```

### 11.2.4 Unsloth 文档

#### 11.2.4.1 基本信息

| 项目 | 内容                     |
| ---- | ------------------------ |
| 链接 | https://docs.unsloth.ai/ |
| 内容 | 加速微调的完整文档       |
| 特点 | 速度快、显存低、易于使用 |

#### 11.2.4.2 核心内容

| 章节     | 内容               |
| -------- | ------------------ |
| 快速入门 | 安装、基本用法     |
| 模型支持 | 支持的模型列表     |
| 微调指南 | LoRA、QLoRA 的用法 |
| 性能优化 | 加速技巧           |
| 常见问题 | FAQ                |

#### 11.2.4.3 推荐学习路径

```
1. 阅读快速入门，完成第一个加速微调
2. 阅读模型支持，确认自己的模型是否支持
3. 阅读微调指南，学习最佳实践
4. 参考性能优化章节，进一步加速
```

## 11.3 综述论文

### 11.3.1 Fine-tuning Large Language Models with Limited Data: A Survey and Practical Guide

#### 11.3.1.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | Fine-tuning Large Language Models with Limited Data: A Survey and Practical Guide |
| 作者 | 多位作者                                                     |
| 发表 | TACL 2026                                                    |
| 链接 | 待补充                                                       |

#### 11.3.1.2 核心内容

| 章节         | 内容                            |
| ------------ | ------------------------------- |
| 数据稀缺问题 | 定义、影响、挑战                |
| 数据增强     | 同义改写、回译、合成数据        |
| PEFT 方法    | LoRA、Adapter、Prefix Tuning 等 |
| 少样本学习   | Few-shot、Zero-shot             |
| 评估方法     | 小数据下的评估策略              |
| 实践指南     | 数据准备、方法选择、超参数调优  |

#### 11.3.1.3 关键发现

| 发现                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| 数据质量 > 数据数量 | 1000 条高质量数据优于 10000 条低质量数据                     |
| 架构依赖的样本效率  | Encoder 模型上 PEFT 在 700 条以下优于全参，Decoder 模型上全参始终更优 |
| 数据多样性关键      | 在单一来源的小数据集上，质量筛选更有效；在大规模数据上，多样性更关键 |

### 11.3.2 Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment

#### 11.3.2.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment |
| 作者 | 多位作者                                                     |
| 发表 | IEEE 2026                                                    |
| 链接 | 待补充                                                       |

#### 11.3.2.2 核心内容

| 章节      | 内容                               |
| --------- | ---------------------------------- |
| PEFT 分类 | 加性、选择性、重参数化、混合、统一 |
| 方法对比  | 各方法的性能、参数量、推理开销     |
| 基准测试  | PEFT-Bench 基准结果                |
| 理论分析  | 表达能力、样本效率                 |
| 实践建议  | 方法选择决策表                     |

#### 11.3.2.3 关键发现

| 发现                 | 说明                                 |
| -------------------- | ------------------------------------ |
| LoRA 综合最优        | 在 PEFT-Bench 上平均性能最高（80.1） |
| BitFit 性价比高      | 参数量仅 0.1%，平均性能 75.3         |
| Prefix Tuning 性能低 | 平均性能仅 45.9，但参数效率极高      |
| 推理开销             | LoRA、BitFit、IA³ 无额外推理开销     |

### 11.3.3 A Technical Survey of Reinforcement Learning Techniques for Large Language Models

#### 11.3.3.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | A Technical Survey of Reinforcement Learning Techniques for Large Language Models |
| 作者 | 多位作者                                                     |
| 发表 | ACM 2026                                                     |
| 链接 | 待补充                                                       |

#### 11.3.3.2 核心内容

| 章节        | 内容                   |
| ----------- | ---------------------- |
| RLHF        | 三阶段流程、PPO 算法   |
| DPO         | 直接偏好优化、数学推导 |
| GRPO        | 组相对策略优化         |
| RLAIF       | AI 反馈强化学习        |
| RLVR        | 可验证奖励强化学习     |
| 多智能体 RL | MARL 框架              |

#### 11.3.3.3 关键发现

| 发现                      | 说明                             |
| ------------------------- | -------------------------------- |
| 对齐方法演进              | PPO → DPO → GRPO → RLVR/RLAIF    |
| GRPO 节省显存             | 省去 Critic 网络，显存减少约 50% |
| RLVR 在推理任务上效果显著 | 数学、代码任务上表现突出         |
| 多智能体 RL 是未来方向    | 处理更复杂的协作任务             |

### 11.3.4 Survey on Joint Compression and Fine-tuning of Large Language Models

#### 11.3.4.1 论文信息

| 项目 | 内容                                                         |
| ---- | ------------------------------------------------------------ |
| 标题 | Survey on joint compression and fine-tuning of large language models |
| 作者 | 多位作者                                                     |
| 发表 | Neurocomputing 2026                                          |
| 链接 | 待补充                                                       |

#### 11.3.4.2 核心内容

| 章节     | 内容                   |
| -------- | ---------------------- |
| 压缩方法 | 剪枝、量化、蒸馏       |
| 微调方法 | 全参、PEFT             |
| 联合优化 | 压缩+微调的统一流水线  |
| 实践指南 | 压缩率、微调策略的选择 |

#### 11.3.4.3 关键发现

| 发现          | 说明                                 |
| ------------- | ------------------------------------ |
| 先压缩再微调  | 降低计算成本，缩短训练时间           |
| 压缩感知微调  | 在微调中同时优化压缩约束             |
| 知识蒸馏+微调 | 先微调大模型作为教师，再蒸馏到小模型 |

### 11.3.5 综述论文对比

| 综述                               | 发表                | 核心主题      | 适合读者               |
| ---------------------------------- | ------------------- | ------------- | ---------------------- |
| Fine-tuning LLMs with Limited Data | TACL 2026           | 小数据微调    | 数据有限的研究者       |
| PEFT Methods Critical Review       | IEEE 2026           | PEFT 方法对比 | 选择 PEFT 方法的实践者 |
| RL Techniques for LLMs             | ACM 2026            | 强化学习对齐  | 对齐研究的入门者       |
| Joint Compression and Fine-tuning  | Neurocomputing 2026 | 压缩+微调     | 部署优化的工程师       |

## 11.4 学习路径建议

### 11.4.1 入门阶段（1-2 周）

| 周次    | 学习内容                           | 实践任务                       |
| ------- | ---------------------------------- | ------------------------------ |
| 第 1 周 | 阅读 LoRA 论文 + PEFT 文档快速入门 | 用 LoRA 微调一个 7B 模型       |
| 第 2 周 | 阅读 QLoRA 论文 + Unsloth 文档     | 用 QLoRA 在单卡上微调 13B 模型 |

### 11.4.2 进阶阶段（2-3 周）

| 周次    | 学习内容                       | 实践任务                    |
| ------- | ------------------------------ | --------------------------- |
| 第 3 周 | 阅读 DPO 论文 + TRL 文档       | 用 DPO 训练一个偏好对齐模型 |
| 第 4 周 | 阅读 GRPO 论文 + DeepSeekMath  | 用 GRPO 训练数学推理模型    |
| 第 5 周 | 阅读 DoRA 论文 + PEFT 高级用法 | 对比 LoRA 和 DoRA 的效果    |

### 11.4.3 实战阶段（3-4 周）

| 周次    | 学习内容                          | 实践任务                        |
| ------- | --------------------------------- | ------------------------------- |
| 第 6 周 | 阅读 InstructGPT 论文 + RLHF 综述 | 搭建完整的 RLHF 流程            |
| 第 7 周 | 阅读 LLaMA-Factory 文档 + 实战    | 完成一个领域微调项目            |
| 第 8 周 | 阅读压缩+微调综述                 | 量化部署微调后的模型            |
| 第 9 周 | 综合实战                          | 端到端：数据准备→微调→评估→部署 |

### 11.4.4 前沿跟踪（持续）

| 活动       | 频率   | 说明                       |
| ---------- | ------ | -------------------------- |
| 关注 arXiv | 每周   | 搜索 cs.CL、cs.LG 最新论文 |
| 关注会议   | 每季度 | ICLR、NeurIPS、ACL、EMNLP  |
| 参与开源   | 持续   | 贡献代码、复现论文         |
| 阅读综述   | 每半年 | 跟踪最新综述论文           |




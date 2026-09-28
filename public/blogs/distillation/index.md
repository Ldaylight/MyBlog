# 一、知识蒸馏基础认知

## 1.1 什么是知识蒸馏

### 1.1.1 定义

知识蒸馏（Knowledge Distillation, KD）是一种模型压缩与知识迁移技术，通过构建"教师-学生"框架，将大型复杂模型（教师模型）中学习到的知识迁移到小型轻量模型（学生模型）中，使学生在显著降低参数量和计算复杂度的同时，尽可能保留教师模型的性能。

![image-20260922170221334](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260922170221334.png)

**形式化定义**：

设教师模型为 $f_T(\cdot; \theta_T)$，学生模型为 $f_S(\cdot; \theta_S)$，其中 $\theta_T$ 和 $\theta_S$ 分别是教师和学生的参数，且 $|\theta_S| \ll |\theta_T|$。知识蒸馏的目标是找到一个学生参数 $\theta_S^*$，使得学生模型在目标任务上的性能尽可能接近教师模型：

$$
\theta_S^* = \arg\min_{\theta_S} \ \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \mathcal{L}_{\text{KD}}\left( f_S(x; \theta_S), f_T(x; \theta_T) \right) + \lambda \mathcal{L}_{\text{task}}\left( f_S(x; \theta_S), y \right) \right]
$$

其中：

- $\mathcal{D}$：训练数据集
- $\mathcal{L}_{\text{KD}}$：蒸馏损失，衡量学生与教师输出的差异
- $\mathcal{L}_{\text{task}}$：任务损失，衡量学生与真实标签的差异
- $\lambda$：平衡系数

### 1.1.2 核心思想

知识蒸馏的核心思想是：**教师模型在训练过程中不仅学到了输入到输出的映射关系，还学到了类别之间的相对关系（即"暗知识"，Dark Knowledge）**。这些暗知识以软标签（Soft Targets）的形式存在，比硬标签（Hard Labels，即 one-hot 编码）包含更丰富的信息。

**硬标签 vs 软标签**：

假设一个手写数字识别任务，输入是一张写着"7"的图片。

**硬标签**（Hard Labels）：
$$
\mathbf{y} = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
$$

即"7"的概率为 1，其余为 0。这种表示丢失了类别之间的相似性信息。

**软标签**（Soft Targets）：
$$
\mathbf{p}_T = [0.01, 0.02, 0.01, 0.05, 0.01, 0.01, 0.02, 0.82, 0.03, 0.02]
$$

即"7"的概率为 0.82，但"3"有 0.05 的概率，"9"有 0.03 的概率，"1"和"6"各 0.02 的概率。这些非目标类的概率反映了**类别之间的视觉相似性**："7"与"3"、"9"、"1"在手写体中有一定的相似性。这种"暗知识"是硬标签无法提供的。

如：

![image-20260922174001223](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260922174001223.png)

比如第一个图片，他是2，但也有点像3。第二个图，它是2，但也有点像7

暗知识无法反映与其他类别的相似性，但是软知识可以

**学生如何从软标签中学习**：

学生模型通过模仿教师的软输出分布，不仅学到"这个样本是 7"，还学到"7 与 3、9、1 更相似，与 0、4、8 差异更大"。这种类别间的关系知识帮助学生模型在有限容量下更高效地学习。

### 1.1.3 知识蒸馏与模型压缩的关系

知识蒸馏是模型压缩的重要技术之一，与剪枝（Pruning）、量化（Quantization）并列。

**模型压缩技术对比**：

|     技术     |           原理            |            优点            |          缺点          | 典型压缩率 |
| :----------: | :-----------------------: | :------------------------: | :--------------------: | :--------: |
|   知识蒸馏   |     师生框架迁移知识      | 灵活、通用，学生架构可不同 | 需要教师模型，训练复杂 |   2-10×    |
|     剪枝     |     移除冗余参数/结构     |   简单直接，可保持稀疏性   | 可能损失性能，需要微调 |   2-10×    |
|     量化     | 降低参数精度（FP32→INT8） |   硬件加速明显，通用性好   |   精度损失，需要校准   |     4×     |
|   低秩分解   |  分解权重矩阵为低秩乘积   |     数学优雅，压缩率高     |  适用性有限，实现复杂  |    2-5×    |
| 紧凑架构设计 |   设计更高效的网络结构    |     从源头优化，效果好     |    需要架构设计经验    |   2-10×    |

**知识蒸馏的独特优势**：

1. **不依赖特定硬件**：蒸馏后的模型可以直接在通用硬件上运行，无需特殊加速器。
2. **可迁移性**：可以将多个模型的知识融合到一个学生模型中。
3. **灵活性**：学生模型架构可以与教师不同（如 CNN → Transformer，或反之）。
4. **可与其它技术组合**：蒸馏可以与剪枝、量化等技术结合使用，进一步提升压缩率。
5. **知识融合**：多教师蒸馏可以将多个专家的知识融合到一个学生中。

## 1.2 为什么需要知识蒸馏

### 1.2.1 模型部署的挑战

#### 1.2.1.1 大模型的资源瓶颈

大模型（如 GPT-4、LLaMA-70B）虽然在各类任务上表现优异，但参数量巨大（数十亿到数千亿），推理时需要大量 GPU 显存，延迟高，难以在移动端、边缘设备或实时系统中部署。

|    模型    | 参数量 | 推理显存（FP16） | 推理延迟（单次） |   硬件需求    |
| :--------: | :----: | :--------------: | :--------------: | :-----------: |
|   GPT-4    | ~1.8T  |      ~3.6TB      |       ~10s       |   多卡 A100   |
| LLaMA-70B  |  70B   |      140GB       |       ~2s        |  2×A100 80GB  |
|  LLaMA-7B  |   7B   |       14GB       |      ~0.2s       | 单卡 RTX 4090 |
| DistilBERT |  66M   |      132MB       |      ~0.01s      |  CPU 可运行   |

#### 1.2.1.2 边缘设备的限制

移动端、IoT 设备、嵌入式系统的计算资源和内存非常有限：

- 手机：通常 4-8GB RAM，无独立 GPU。
- 智能手表：通常 512MB-1GB RAM。
- 嵌入式芯片：通常 256MB-2GB RAM。

这些设备无法直接运行大模型，需要将模型压缩到适合部署的大小。

#### 1.2.1.3 实时性要求

某些应用对延迟极度敏感：

- 自动驾驶：决策延迟 < 100ms。
- 实时翻译：延迟 < 500ms。
- 在线推荐：延迟 < 50ms。

大模型的推理延迟难以满足这些要求。

![image-20260922172041845](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260922172041845.png)

### 1.2.2 知识蒸馏的核心价值

应用场景：

![img](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typora8e0df1c1d4dcb0c635a47b9a7ca35ea2_720.jpg)

|  价值维度  |                   具体体现                   |         量化示例         |
| :--------: | :------------------------------------------: | :----------------------: |
|  模型压缩  |          学生模型参数量可减少数十倍          | DistilBERT 减少 40% 参数 |
|  性能保留  |    学生模型可达到教师模型 90%-99% 的性能     | DistilBERT 保留 97% 性能 |
|  推理加速  |           学生模型推理速度提升数倍           | DistilBERT 速度提升 60%  |
|  知识迁移  |         将教师模型的暗知识传递给学生         |       类别关系知识       |
|  集成学习  |       多教师蒸馏可将多个模型的知识融合       |        多专家融合        |
|   自改进   |        自蒸馏可提升模型自身的泛化能力        |   Born-Again Networks    |
|  隐私保护  | 无数据蒸馏可在不访问原始数据的情况下迁移知识 |      医疗、金融场景      |
| 跨模态迁移 |                不同模态间蒸馏                |        视觉→文本         |

### 1.2.3 知识蒸馏 vs 其他压缩技术

#### 1.2.3.1 知识蒸馏 vs 剪枝

|      维度       |     知识蒸馏     |         剪枝         |
| :-------------: | :--------------: | :------------------: |
|      原理       | 师生框架迁移知识 |     移除冗余参数     |
| 学生/剪枝后模型 |  可以是全新架构  |    通常保持原架构    |
|    训练方式     |   需要教师指导   |     通常需要微调     |
|     压缩率      |      2-10×       |        2-10×         |
|    性能损失     | 小（有教师指导） | 中等（需要微调恢复） |
|     灵活性      | 高（架构可不同） |   低（保持原架构）   |

#### 1.2.3.2 知识蒸馏 vs 量化

|   维度   |     知识蒸馏     |       量化       |
| :------: | :--------------: | :--------------: |
|   原理   | 迁移知识到小模型 |   降低参数精度   |
| 输出模型 |    浮点小模型    |    低精度模型    |
| 硬件需求 |     通用硬件     | 需要量化加速支持 |
|  压缩率  |      2-10×       |    4×（INT8）    |
| 性能损失 |        小        | 中等（需要校准） |
| 组合使用 |   可与量化结合   |   可与蒸馏结合   |

#### 1.2.3.3 知识蒸馏 + 剪枝 + 量化的组合

三种技术可以组合使用，实现更高的压缩率：

```
原始模型 → 知识蒸馏（架构压缩）→ 剪枝（参数压缩）→ 量化（精度压缩）→ 部署
```

例如：

- 知识蒸馏：70B → 7B（参数量减少 10×）
- 剪枝：7B → 5B（稀疏化 30%）
- 量化：5B → 2.5B（INT8 量化 4×）

最终压缩率：70B → 2.5B，约 28×。

## 1.3 知识蒸馏的发展历程

### 1.3.1 起源（2015）

#### 1.3.1.1 Hinton 的开创性工作

Hinton 等人于 2015 年提出知识蒸馏的经典框架，论文《Distilling the Knowledge in a Neural Network》奠定了该领域的基础。

**核心贡献**：

1. **引入温度参数**：在 Softmax 中引入温度 $T$，使教师的输出分布更平滑，暴露更多暗知识。
2. **软标签蒸馏**：学生模仿教师的软输出，而非硬标签。
3. **MNIST 实验**：在 MNIST 上验证了蒸馏的有效性。教师模型是一个大型集成模型，学生模型是一个小型的单层网络。
4. **语音识别实验**：在语音识别任务上，蒸馏后的模型显著优于直接用硬标签训练的模型。

**历史意义**：

- 首次系统性地提出知识蒸馏的概念。
- 温度参数和软标签成为后续所有蒸馏方法的基础。
- 启发了大量后续研究。

### 1.3.2 发展期（2016-2020）

#### 1.3.2.1 FitNets（2015）

Romero 等人提出 FitNets，首次将蒸馏从输出层扩展到中间层。

**核心贡献**：

- 引入 Hint 层：教师中间层的特征作为"提示"，指导学生中间层的学习。
- 解决了深度网络难以训练的问题：通过 Hint 层，学生可以更好地学习深层表示。

**数学表达**：

$$
\mathcal{L}_{\text{fitnet}} = \| f_{\text{student}}(x) - \phi(f_{\text{teacher}}(x)) \|^2
$$

其中 $\phi$ 是维度对齐函数（通常是一个 1×1 卷积或全连接层）。

#### 1.3.2.2 Attention Transfer（2017）

Zagoruyko 和 Komodakis 提出 Attention Transfer，将教师的注意力图迁移给学生。

**核心贡献**：

- 使用注意力图（Attention Map）作为知识表示。
- 注意力图是特征图在通道维度上的统计量。

**数学表达**：

$$
\mathcal{L}_{\text{AT}} = \left\| \frac{\mathbf{A}_S}{\|\mathbf{A}_S\|_2} - \frac{\mathbf{A}_T}{\|\mathbf{A}_T\|_2} \right\|_2
$$

其中 $\mathbf{A} = \sum_{c} |\mathbf{F}_c|^2$ 是注意力图。

#### 1.3.2.3 RKD（2019）

Park 等人提出 Relational Knowledge Distillation，关注样本之间的关系。

**核心贡献**：

- 将知识表示为样本之间的关系，而非单个样本的特征。
- 使用距离和角度作为关系度量。

**数学表达**：

$$
\mathcal{L}_{\text{RKD-D}} = \sum_{(x_i, x_j)} \| d(f_S(x_i), f_S(x_j)) - d(f_T(x_i), f_T(x_j)) \|^2
$$

$$
\mathcal{L}_{\text{RKD-A}} = \sum_{(x_i, x_j, x_k)} \| \angle(f_S(x_i), f_S(x_j), f_S(x_k)) - \angle(f_T(x_i), f_T(x_j), f_T(x_k)) \|^2
$$

#### 1.3.2.4 CRD（2020）

Tian 等人提出 Contrastive Representation Distillation，将对比学习引入蒸馏。

**核心贡献**：

- 使用对比学习框架，拉近学生与教师正样本的距离，推远负样本的距离。
- 在多个基准上取得最优性能。

**数学表达**：

$$
\mathcal{L}_{\text{CRD}} = -\mathbb{E}_{(x, y)} \left[ \log \frac{\exp(\text{sim}(f_S(x), f_T(x)) / \tau)}{\sum_{x'} \exp(\text{sim}(f_S(x), f_T(x')) / \tau)} \right]
$$

#### 1.3.2.5 DML（2018）

Zhang 等人提出 Deep Mutual Learning，多个模型同时训练，互为师生。

**核心贡献**：

- 不需要预训练教师，多个学生模型互相学习。
- 证明了互相学习可以提升所有模型的性能。

**数学表达**：

$$
\mathcal{L}_{\text{DML}} = \sum_{i} \sum_{j \neq i} \text{KL}(p_i \parallel p_j) + \sum_{i} \text{CE}(p_i, y)
$$

### 1.3.3 繁荣期（2021至今）

#### 1.3.3.1 大语言模型蒸馏

- **DistilBERT（2019）**：将 BERT 蒸馏为 6 层的小模型，参数量减少 40%，速度提升 60%，保留 97% 的性能。
- **TinyBERT（2020）**：将 BERT 蒸馏为 4 层的小模型，参数量减少 75%，速度提升 9.4 倍。
- **MiniLM（2021）**：蒸馏 BERT 的最后一层自注意力，保留 99% 的性能。
- **LLM 蒸馏（2023-2024）**：将 LLaMA-70B 蒸馏为 7B 模型，保留 90% 以上的性能。

#### 1.3.3.2 解耦知识蒸馏（DKD, 2022）

将蒸馏损失解耦为目标类知识蒸馏（TCKD）和非目标类知识蒸馏（NCKD），分别处理不同来源的知识。

#### 1.3.3.3 无数据蒸馏（Data-Free KD）

在没有原始训练数据的情况下进行蒸馏，适用于隐私敏感场景。

#### 1.3.3.4 跨模态蒸馏

将知识从一个模态（如图像）迁移到另一个模态（如文本）。

#### 1.3.3.5 多教师蒸馏

融合多个教师的知识，学生模型可以综合多个专家的优势。

#### 1.3.3.6 对抗蒸馏

将 GAN 的思想引入蒸馏，通过对抗训练提升蒸馏效果。

#### 1.3.3.7 自蒸馏

模型从自身的早期版本或不同分支中学习，无需额外的教师模型。

#### 1.3.3.8 联邦蒸馏

在联邦学习中，各客户端模型互相蒸馏，保护数据隐私。

#### 1.3.3.9 发展脉络总结

```
2015: Hinton KD（开创）
  ↓
2015-2016: FitNets（特征蒸馏）
  ↓
2017: Attention Transfer（注意力蒸馏）
  ↓
2018: DML（在线蒸馏）
  ↓
2019: RKD（关系蒸馏）
  ↓
2020: CRD（对比蒸馏）
  ↓
2021-2024: LLM 蒸馏、DKD、无数据蒸馏、跨模态蒸馏、多教师蒸馏
  ↓
2025+: 多模态蒸馏、联邦蒸馏、自蒸馏、对抗蒸馏
```

**各阶段核心方法对比**：

| 阶段 |      代表方法      |  知识类型   |    核心创新    |  典型应用  |
| :--- | :----------------: | :---------: | :------------: | :--------: |
| 起源 |     Hinton KD      |    响应     | 温度 + 软标签  |  图像分类  |
| 发展 |      FitNets       |    特征     |    Hint 层     |  深度网络  |
| 发展 | Attention Transfer |    特征     |    注意力图    |  图像分类  |
| 发展 |        RKD         |    关系     |    样本关系    |  度量学习  |
| 发展 |        DML         |    响应     |    在线蒸馏    | 半监督学习 |
| 发展 |        CRD         |    关系     |    对比学习    |  表示学习  |
| 繁荣 |     DistilBERT     | 响应 + 特征 |   大模型蒸馏   |    NLP     |
| 繁荣 |        DKD         |    响应     | 解耦 TCKD/NCKD |  图像分类  |
| 繁荣 |     无数据蒸馏     |    响应     |    合成数据    |  隐私场景  |
| 繁荣 |     跨模态蒸馏     |    特征     |    模态对齐    |   多模态   |
| 前沿 |      联邦蒸馏      |    响应     |    联邦学习    |  隐私保护  |



# 二、知识蒸馏核心机制

## 2.1 基础框架：Hinton KD

### 2.1.1 核心思想

Hinton 等人于 2015 年提出的知识蒸馏框架，是知识蒸馏领域的奠基性工作。其核心思想是：**通过温度参数软化教师的输出分布，使学生模型能够从教师的"暗知识"中学习，而不仅仅是从硬标签中学习。**

传统训练中，模型使用硬标签（one-hot 编码）作为监督信号。硬标签只告诉模型"正确答案是什么"，但没有告诉模型"错误答案之间有什么差异"。例如，在手写数字识别中，硬标签只说"这是 7"，但没说"7 更像 1 还是更像 9"。

Hinton 的洞察是：**教师模型的软输出（soft targets）包含了类别之间的相对关系信息，这种信息比硬标签更丰富、更平滑，能够为学生提供更多的学习信号。**

过程如图：

![image-20260923090855634](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260923090855634.png)

![image-20260924081515988](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260924081515988.png)

### 2.1.2 Softmax with Temperature

#### 2.1.2.1 标准 Softmax

给定模型的 logits 向量 $\mathbf{z} = (z_1, z_2, \dots, z_K)$，标准 Softmax 将其转换为概率分布：

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}
$$

其中：

- $z_i$：第 $i$ 类的 logit（未归一化的得分）
- $K$：类别总数
- $p_i$：第 $i$ 类的预测概率

**性质**：

- $p_i \in (0, 1)$
- $\sum_{i=1}^{K} p_i = 1$
- 当某个 $z_i$ 远大于其他 logits 时，$p_i \to 1$，其他 $p_j \to 0$

#### 2.1.2.2 带温度的 Softmax

Hinton 引入温度参数 $T$，将标准 Softmax 推广为：

$$
q_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{K} \exp(z_j / T)}
$$

其中 $T > 0$ 是温度参数。

**温度参数的作用分析**：

**（1）当 $T = 1$ 时**：
$$
q_i = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)} = p_i
$$

退化为标准 Softmax。

**（2）当 $T \to \infty$ 时**：

$$
\lim_{T \to \infty} q_i = \lim_{T \to \infty} \frac{\exp(z_i / T)}{\sum_{j=1}^{K} \exp(z_j / T)} = \frac{1}{K}
$$

分布趋于均匀分布。

**推导**：当 $T \to \infty$ 时，$z_i / T \to 0$，所以 $\exp(z_i / T) \to 1$，因此：

$$
q_i \to \frac{1}{\sum_{j=1}^{K} 1} = \frac{1}{K}
$$

**（3）当 $T \to 0$ 时**：

$$
\lim_{T \to 0} q_i = \begin{cases} 1 & \text{if } z_i = \max_j z_j \\ 0 & \text{otherwise} \end{cases}
$$

分布趋于 one-hot 分布。

**推导**：设 $z_{\max} = \max_j z_j$。当 $T \to 0$ 时，$z_{\max}/T$ 远大于其他 $z_j/T$，因此 $\exp(z_{\max}/T)$ 远大于 $\exp(z_j/T)$，所以 $q_{\max} \to 1$，其他 $q_j \to 0$。

**（4）当 $T > 1$ 时**：

分布变得==更加平滑==，非最大类别的概率被放大，类别之间的相对关系更加明显。

**示例**：

假设 logits 为 $\mathbf{z} = (5, 3, 1)$，类别数为 3。

当 $T = 1$ 时：

$$
p_1 = \frac{e^5}{e^5 + e^3 + e^1} = \frac{148.4}{148.4 + 20.1 + 2.7} = \frac{148.4}{171.2} \approx 0.867
$$

$$
p_2 = \frac{e^3}{171.2} \approx 0.117
$$

$$
p_3 = \frac{e^1}{171.2} \approx 0.016
$$

当 $T = 4$ 时：

$$
q_1 = \frac{e^{5/4}}{e^{5/4} + e^{3/4} + e^{1/4}} = \frac{3.49}{3.49 + 2.12 + 1.28} = \frac{3.49}{6.89} \approx 0.507
$$

$$
q_2 = \frac{2.12}{6.89} \approx 0.308
$$

$$
q_3 = \frac{1.28}{6.89} \approx 0.185
$$

可以看到，温度 $T=4$ 时，非最大类别的概率显著增大（从 0.117 和 0.016 增加到 0.308 和 0.185），类别之间的关系更加明显。

![image-20260923084835604](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260923084835604.png)



### 2.1.3 蒸馏损失函数

#### 2.1.3.1 总损失函数

知识蒸馏的总损失由两部分组成：

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{KD}} + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}
$$

其中：

- $\mathcal{L}_{\text{KD}}$：蒸馏损失，学生模仿教师的软输出
- $\mathcal{L}_{\text{CE}}$：学生与真实标签的交叉熵损失
- $\alpha$：平衡系数，通常取 0.5-0.9

#### 2.1.3.2 蒸馏损失

蒸馏损失使用 KL 散度（Kullback-Leibler Divergence）衡量学生分布与教师分布的差异：

$$
\mathcal{L}_{\text{KD}} = \text{KL}\left( \mathbf{q}^T \parallel \mathbf{q}^S \right) = \sum_{i=1}^{K} q_i^T \log \frac{q_i^T}{q_i^S}
$$

其中：

- $q_i^T = \frac{\exp(z_i^T / T)}{\sum_j \exp(z_j^T / T)}$：教师的软输出
- $q_i^S = \frac{\exp(z_i^S / T)}{\sum_j \exp(z_j^S / T)}$：学生的软输出

**展开 KL 散度**：

$$
\mathcal{L}_{\text{KD}} = \sum_{i=1}^{K} q_i^T \log q_i^T - \sum_{i=1}^{K} q_i^T \log q_i^S
$$

第一项 $\sum_i q_i^T \log q_i^T$ 是教师的熵，对于固定的教师是常数，与学生的参数无关。因此，最小化 KL 散度等价于最小化交叉熵：

$$
\mathcal{L}_{\text{KD}} = -\sum_{i=1}^{K} q_i^T \log q_i^S + \text{const}
$$

因此，蒸馏损失可以简化为：

$$
\mathcal{L}_{\text{KD}} = -\sum_{i=1}^{K} q_i^T \log q_i^S
$$

**注意**：Hinton 在论文中使用了 $T^2$ 缩放：

$$
\mathcal{L}_{\text{KD}} = T^2 \cdot \text{KL}\left( \mathbf{q}^T \parallel \mathbf{q}^S \right)
$$

**为什么要乘 $T^2$？**

当 $T$ 较大时，梯度会变小（见下文梯度分析），乘以 $T^2$ 可以保持梯度与 $T=1$ 时相当的量级。具体推导见 2.1.4 节。

#### 2.1.3.3 交叉熵损失

学生与真实标签的交叉熵损失为：

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{K} y_i \log p_i^S
$$

其中：

- $y_i$：真实标签的 one-hot 编码
- $p_i^S = \frac{\exp(z_i^S)}{\sum_j \exp(z_j^S)}$：学生的标准 Softmax 输出（温度 $T=1$）

#### 2.1.3.4 完整损失函数

$$
\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\left( \mathbf{q}^T \parallel \mathbf{q}^S \right) + (1 - \alpha) \cdot \text{CE}(\mathbf{p}^S, \mathbf{y})
$$

计算过程图示：

![image-20260924091512023](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260924091512023.png)

### 2.1.4 梯度分析

#### 2.1.4.1 蒸馏损失的梯度推导

**（1）对 logits 的梯度**

设学生的 logits 为 $z_i^S$，蒸馏损失为：

$$
\mathcal{L}_{\text{KD}} = -\sum_{j=1}^{K} q_j^T \log q_j^S
$$

其中 $q_j^S = \frac{\exp(z_j^S / T)}{\sum_k \exp(z_k^S / T)}$。

对 $z_i^S$ 求偏导：

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S} = -\sum_{j=1}^{K} q_j^T \frac{\partial \log q_j^S}{\partial z_i^S}
$$

计算 $\frac{\partial \log q_j^S}{\partial z_i^S}$：

$$
\log q_j^S = \frac{z_j^S}{T} - \log \sum_{k=1}^{K} \exp\left( \frac{z_k^S}{T} \right)
$$

$$
\frac{\partial \log q_j^S}{\partial z_i^S} = \frac{\delta_{ij}}{T} - \frac{1}{T} \cdot \frac{\exp(z_i^S / T)}{\sum_k \exp(z_k^S / T)} = \frac{\delta_{ij} - q_i^S}{T}
$$

其中 $\delta_{ij}$ 是 Kronecker delta。

代入：

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S} = -\sum_{j=1}^{K} q_j^T \cdot \frac{\delta_{ij} - q_i^S}{T}
$$

$$
= -\frac{1}{T} \sum_{j=1}^{K} q_j^T \delta_{ij} + \frac{q_i^S}{T} \sum_{j=1}^{K} q_j^T
$$

由于 $\sum_j q_j^T = 1$：

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S} = -\frac{q_i^T}{T} + \frac{q_i^S}{T} = \frac{q_i^S - q_i^T}{T}
$$

**结论**：

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S} = \frac{1}{T} (q_i^S - q_i^T)
$$

**直观理解**：梯度的方向是让学生的软输出 $q_i^S$ 靠近教师的软输出 $q_i^T$。如果 $q_i^S > q_i^T$，梯度为正，更新会减小 $z_i^S$；如果 $q_i^S < q_i^T$，梯度为负，更新会增大 $z_i^S$。

#### 2.1.4.2 高温下的近似

当 $T$ 很大时，$z_i / T$ 很小，可以使用泰勒展开：

$$
\exp\left( \frac{z_i}{T} \right) \approx 1 + \frac{z_i}{T} + \frac{1}{2} \left( \frac{z_i}{T} \right)^2 + \dots
$$

保留一阶项：

$$
q_i = \frac{1 + z_i / T}{\sum_j (1 + z_j / T)} = \frac{1 + z_i / T}{K + \sum_j z_j / T}
$$

假设 $\sum_j z_j = 0$（logits 通常零均值化），则：

$$
q_i \approx \frac{1 + z_i / T}{K} = \frac{1}{K} + \frac{z_i}{KT}
$$

因此：

$$
q_i^S - q_i^T \approx \frac{z_i^S - z_i^T}{KT}
$$

代入梯度公式：

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S} \approx \frac{1}{T} \cdot \frac{z_i^S - z_i^T}{KT} = \frac{z_i^S - z_i^T}{KT^2}
$$

**结论**：在高温下，蒸馏损失的梯度近似为 logits 之差的缩放：

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S} \approx \frac{z_i^S - z_i^T}{KT^2}
$$

这意味着，**蒸馏损失在高温下等价于 logits 之间的均方误差（MSE）**：

$$
\mathcal{L}_{\text{KD}} \approx \frac{1}{2KT^2} \| \mathbf{z}^S - \mathbf{z}^T \|^2
$$

**为什么乘 $T^2$？**

从上面的推导可以看到，梯度中出现了 $1/T^2$ 的因子。当 $T$ 较大时，梯度会变得很小，导致训练缓慢。乘以 $T^2$ 后：

$$
T^2 \cdot \frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_i^S} \approx \frac{z_i^S - z_i^T}{K}
$$

梯度与 $T$ 无关，保持与 $T=1$ 时相当的量级。

#### 2.1.4.3 完整损失函数的梯度

$$
\frac{\partial \mathcal{L}}{\partial z_i^S} = \alpha \cdot T^2 \cdot \frac{1}{T} (q_i^S - q_i^T) + (1 - \alpha) \cdot (p_i^S - y_i)
$$

$$
= \alpha \cdot T (q_i^S - q_i^T) + (1 - \alpha) \cdot (p_i^S - y_i)
$$

其中 $p_i^S$ 是标准 Softmax（$T=1$）的输出。

**分析**：

- 第一项：蒸馏损失梯度，推动学生软输出靠近教师软输出。
- 第二项：交叉熵梯度，推动学生硬输出靠近真实标签。
- $\alpha$ 控制两者的平衡。

### 2.1.5 温度参数的作用

#### 2.1.5.1 温度对暗知识的影响

**（1）低温度（$T = 1$）**

教师的输出分布接近 one-hot，非目标类的概率很小，暗知识被"淹没"在接近零的概率中。

例如，教师输出为 $(0.98, 0.01, 0.005, 0.003, 0.002)$，非目标类概率很小，学生很难从中学到类别关系。

**（2）高温度（$T = 4$）**

教师的输出分布更平滑，非目标类的概率被放大，暗知识更明显。

例如，温度 $T=4$ 时，教师输出可能变为 $(0.6, 0.2, 0.1, 0.06, 0.04)$，学生可以清晰地看到类别之间的关系。

#### 2.1.5.2 温度选择的经验法则

| 教师-学生能力差距 |     推荐温度     |      说明      |
| :---------------: | :--------------: | :------------: |
|      差距小       |  $T = 1 \sim 2$  | 不需要过多平滑 |
|     差距中等      |  $T = 3 \sim 5$  |    常规设置    |
|      差距大       | $T = 5 \sim 10$  | 需要更多暗知识 |
|     极小模型      | $T = 10 \sim 20$ |    极大平滑    |

#### 2.1.5.3 温度与 alpha 的联合调优

经验法则：

- 高温（$T > 5$）时，$\alpha$ 可设为 0.9-0.95（更多依赖蒸馏损失）。
- 低温（$T < 3$）时，$\alpha$ 可设为 0.5-0.7（更多依赖真实标签）。

### 2.1.6 代码实现

#### 2.1.6.1 蒸馏损失函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """
    知识蒸馏损失函数
    
    参数:
    - student_logits: 学生模型的 logits, 形状 [batch, num_classes]
    - teacher_logits: 教师模型的 logits, 形状 [batch, num_classes]
    - labels: 真实标签, 形状 [batch]
    - T: 温度参数
    - alpha: 蒸馏损失权重
    
    返回:
    - total_loss: 总损失
    - kd_loss: 蒸馏损失
    - ce_loss: 交叉熵损失
    """
    # 1. 蒸馏损失：学生模仿教师的软输出
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    
    # 2. 交叉熵损失：学生与真实标签
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # 3. 总损失
    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
    
    return total_loss, kd_loss, ce_loss
```

#### 2.1.6.2 完整的训练循环

```python
def train_student(teacher_model, student_model, train_loader, optimizer, 
                  epochs=10, T=4.0, alpha=0.7):
    """
    训练学生模型
    """
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_kd = 0
        total_ce = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()
            
            # 教师前向传播（不计算梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            # 学生前向传播
            student_logits = student_model(data)
            
            # 计算蒸馏损失
            loss, kd_loss, ce_loss = distillation_loss(
                student_logits, teacher_logits, labels, T, alpha
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_kd += kd_loss.item()
            total_ce += ce_loss.item()
        
        n = len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/n:.4f} | "
              f"KD: {total_kd/n:.4f} | "
              f"CE: {total_ce/n:.4f}")
```

## 2.2 知识类型分类

### 2.2.1 基于响应的蒸馏（Response-based KD）

#### 2.2.1.1 定义

基于响应的蒸馏是指学生模型模仿教师模型最后一层的输出（logits 或软标签）。这是最基础、最通用的蒸馏方式。

#### 2.2.1.2 核心思想

教师的输出包含了类别之间的相对关系信息。例如，在手写数字识别中，教师对"7"的预测可能同时给"1"和"9"一定的概率，这些概率反映了类别之间的视觉相似性。

**数学表达**：

$$
\mathcal{L}_{\text{response}} = \text{KL}\left( \sigma\left( \frac{\mathbf{z}^T}{T} \right) \parallel \sigma\left( \frac{\mathbf{z}^S}{T} \right) \right)
$$

其中 $\sigma$ 是 Softmax 函数。

#### 2.2.1.3 优点与缺点

**优点**：

- 简单、通用，不需要访问教师内部结构。
- 适用于任何模型架构。
- 计算开销低。

**缺点**：

- 只利用了最后一层的信息，忽略了中间层的丰富表示。
- 对于复杂任务，最后一层的知识可能不足以指导学生。

#### 2.2.1.4 代表方法

Hinton KD（2015）。

### 2.2.2 基于特征的蒸馏（Feature-based KD）

#### 2.2.2.1 定义

基于特征的蒸馏是指学生模型模仿教师模型中间层的特征表示。

#### 2.2.2.2 核心思想

深层网络的中间层学到了层次化的特征表示，从低级边缘到高级语义。学生通过模仿这些中间特征，可以学到更丰富的表示。

**数学表达**：

$$
\mathcal{L}_{\text{feature}} = \left\| f_{\text{student}}(x) - \phi(f_{\text{teacher}}(x)) \right\|^2
$$

其中：

- $f_{\text{student}}(x)$：学生中间层的特征
- $f_{\text{teacher}}(x)$：教师中间层的特征
- $\phi$：维度对齐函数（如 1×1 卷积、全连接层）

#### 2.2.2.3 维度对齐问题

教师和学生的特征维度可能不同。例如：

- 教师中间层维度：512
- 学生中间层维度：256

需要设计映射函数 $\phi$ 将学生特征映射到教师特征空间：

$$
\phi: \mathbb{R}^{256} \to \mathbb{R}^{512}
$$

常用的映射函数：

- **1×1 卷积**：适用于 CNN 特征图
- **全连接层**：适用于向量特征
- **线性投影**：$\phi(\mathbf{h}) = \mathbf{W} \mathbf{h}$，其中 $\mathbf{W} \in \mathbb{R}^{d_T \times d_S}$

#### 2.2.2.4 优点与缺点

**优点**：

- 利用了中间层的丰富表示。
- 可以指导学生学到更好的特征。

**缺点**：

- 需要访问教师内部结构。
- 需要设计维度对齐函数。
- 计算开销较大。

#### 2.2.2.5 代表方法

- FitNets（2015）：使用 Hint 层。
- Attention Transfer（2017）：使用注意力图。
- PKT（2018）：使用概率分布迁移。

### 2.2.3 基于关系的蒸馏（Relation-based KD）

#### 2.2.3.1 定义

基于关系的蒸馏是指学生模型模仿教师模型中样本之间或层之间的关系。

#### 2.2.3.2 核心思想

知识不仅存在于单个样本的输出中，还存在于样本之间的关系中。例如，教师可能认为样本 A 和样本 B 相似，这种相似性关系也是一种知识。

**数学表达**：

$$
\mathcal{L}_{\text{relation}} = \left\| \psi\left( f_{\text{student}}(x_i), f_{\text{student}}(x_j) \right) - \psi\left( f_{\text{teacher}}(x_i), f_{\text{teacher}}(x_j) \right) \right\|^2
$$

其中 $\psi$ 是关系函数（如距离、角度、相似度）。

#### 2.2.3.3 关系类型

**（1）距离关系（Distance-wise）**

$$
\psi_{\text{distance}}(f(x_i), f(x_j)) = \| f(x_i) - f(x_j) \|_2
$$

学生模仿教师中样本之间的距离：

$$
\mathcal{L}_{\text{RKD-D}} = \sum_{(x_i, x_j)} \left\| d_S(x_i, x_j) - d_T(x_i, x_j) \right\|^2
$$

其中 $d_S(x_i, x_j) = \| f_S(x_i) - f_S(x_j) \|_2$，$d_T(x_i, x_j) = \| f_T(x_i) - f_T(x_j) \|_2$。

**（2）角度关系（Angle-wise）**

$$
\psi_{\text{angle}}(f(x_i), f(x_j), f(x_k)) = \cos \angle f(x_i) f(x_j) f(x_k)
$$

学生模仿教师中三元组之间的角度：

$$
\mathcal{L}_{\text{RKD-A}} = \sum_{(x_i, x_j, x_k)} \left\| \cos \angle_S - \cos \angle_T \right\|^2
$$

其中：

$$
\cos \angle_S = \frac{\langle f_S(x_i) - f_S(x_j), f_S(x_k) - f_S(x_j) \rangle}{\| f_S(x_i) - f_S(x_j) \| \cdot \| f_S(x_k) - f_S(x_j) \|}
$$

**（3）相似度关系**

$$
\psi_{\text{similarity}}(f(x_i), f(x_j)) = \text{cosine\_sim}(f(x_i), f(x_j))
$$

#### 2.2.3.4 优点与缺点

**优点**：

- 利用了样本之间的结构知识。
- 不要求学生和教师的特征维度相同。
- 对架构差异更鲁棒。

**缺点**：

- 计算开销大（需要计算样本对或三元组）。
- 实现复杂。
- 需要批量大小足够大。

#### 2.2.3.5 代表方法

- RKD（2019）：距离和角度关系。
- CRD（2020）：对比学习关系。
- SP（2020）：相似度保持。

### 2.2.4 补充：基于架构的蒸馏

#### 2.2.4.1 定义

基于架构的蒸馏是指学生模型模仿教师模型的特定架构组件或连接模式。

#### 2.2.4.2 核心思想

教师模型的某些架构组件（如注意力模块、残差连接）是其性能的关键。学生通过模仿这些组件的输入输出关系，可以学到类似的表示。

**代表方法**：

- **TinyBERT（2020）**：学生模仿教师的嵌入层、注意力矩阵、隐藏状态和预测层。
- **MiniLM（2021）**：学生模仿教师的最后一层自注意力。

**数学表达**：

$$
\mathcal{L}_{\text{arch}} = \sum_{l} \left\| \text{Attn}_S^{(l)} - \text{Attn}_T^{(l)} \right\|^2
$$

其中 $\text{Attn}^{(l)}$ 是第 $l$ 层的注意力矩阵。

## 2.3 三种知识类型的对比

### 2.3.1 对比表格

|         维度         |   响应蒸馏   |   特征蒸馏   |   关系蒸馏    |
| :------------------: | :----------: | :----------: | :-----------: |
|       知识来源       | 最后一层输出 |  中间层特征  | 样本/层间关系 |
|        信息量        |     较少     |     丰富     |    最丰富     |
|       实现难度       |     简单     |     中等     |     复杂      |
|       计算开销       |      低      |     中等     |      高       |
| 是否需要访问教师内部 |      否      |      是      |      是       |
|   是否需要维度对齐   |      否      |      是      |      否       |
|  对架构差异的鲁棒性  |      高      |      低      |      高       |
|       适用场景       |     通用     | 需要丰富表示 | 需要结构知识  |
|       代表方法       |  Hinton KD   | FitNets, AT  |   RKD, CRD    |

### 2.3.2 信息量的数学度量

可以用互信息来度量不同知识类型的信息量：

$$
I(T; S) = H(T) - H(T \mid S)
$$

其中：

- $I(T; S)$：教师 $T$ 和学生 $S$ 之间的互信息
- $H(T)$：教师的熵
- $H(T \mid S)$：在已知学生的条件下教师的条件熵

**响应蒸馏**：只利用了最后一层的输出，信息量为：

$$
I_{\text{response}} = I(\mathbf{z}^T; \mathbf{z}^S)
$$

**特征蒸馏**：利用了中间层的特征，信息量为：

$$
I_{\text{feature}} = \sum_{l} I(\mathbf{h}_l^T; \mathbf{h}_l^S)
$$

**关系蒸馏**：利用了样本之间的关系，信息量为：

$$
I_{\text{relation}} = I(\psi(\mathbf{z}_i^T, \mathbf{z}_j^T); \psi(\mathbf{z}_i^S, \mathbf{z}_j^S))
$$

**结论**：关系蒸馏的信息量最大，特征蒸馏次之，响应蒸馏最小。但信息量越大，计算开销也越大，需要根据实际场景权衡。

### 2.3.3 组合使用

三种知识类型可以组合使用：

$$
\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{response}} + \beta \mathcal{L}_{\text{feature}} + \gamma \mathcal{L}_{\text{relation}}
$$

其中 $\alpha, \beta, \gamma$ 是权重系数。

**组合策略**：

| 场景         | 推荐组合           | 说明                 |
| ------------ | ------------------ | -------------------- |
| 资源充足     | 响应 + 特征 + 关系 | 信息最丰富           |
| 资源中等     | 响应 + 特征        | 平衡信息量和开销     |
| 资源有限     | 仅响应             | 简单高效             |
| 架构差异大   | 响应 + 关系        | 对架构差异鲁棒       |
| 需要丰富表示 | 特征 + 关系        | 利用中间层和结构信息 |





# 三、进阶蒸馏技术

## 3.1 离线蒸馏与在线蒸馏

### 3.1.1 离线蒸馏（Offline Distillation）

#### 3.1.1.1 定义

离线蒸馏是指先独立训练好教师模型，然后固定教师模型的参数，用它来指导学生模型的训练。这是最经典、最常用的蒸馏范式。

#### 3.1.1.2 流程

```
步骤1：独立训练教师模型（大规模、高精度）
    ↓
步骤2：冻结教师模型参数
    ↓
步骤3：用教师的软输出指导学生模型训练
    ↓
步骤4：学生模型独立部署
```

**伪代码**：

```python
# 步骤1：训练教师模型
teacher_model = train_teacher(train_dataset, epochs=100)

# 步骤2：冻结教师模型
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# 步骤3：训练学生模型
student_model = StudentModel()
optimizer = AdamW(student_model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for data, labels in train_loader:
        with torch.no_grad():
            teacher_logits = teacher_model(data)
        student_logits = student_model(data)
        loss = distillation_loss(student_logits, teacher_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 3.1.1.3 数学表达

离线蒸馏的优化目标为：

$$
\theta_S^* = \arg\min_{\theta_S} \ \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \alpha \cdot T^2 \cdot \text{KL}\left( \sigma\left( \frac{f_T(x)}{T} \right) \parallel \sigma\left( \frac{f_S(x)}{T} \right) \right) + (1 - \alpha) \cdot \text{CE}\left( f_S(x), y \right) \right]
$$

其中 $\theta_T$ 在训练过程中保持固定。

#### 3.1.1.4 优点与缺点

**优点**：

| 优点       | 说明                         |
| ---------- | ---------------------------- |
| 简单稳定   | 教师已训练好，训练过程可控   |
| 教师质量高 | 可以独立优化教师模型至最优   |
| 计算解耦   | 教师训练和学生训练可以分离   |
| 灵活       | 可以用同一个教师指导多个学生 |

**缺点**：

| 缺点         | 说明                                   |
| ------------ | -------------------------------------- |
| 两阶段训练   | 需要先训练教师，耗时较长               |
| 教师无法更新 | 训练过程中教师参数固定，无法自适应调整 |
| 能力差距固定 | 如果教师和学生差距太大，蒸馏效果受限   |
| 无法协同学习 | 教师和学生不能互相促进                 |

### 3.1.2 在线蒸馏（Online Distillation）

#### 3.1.2.1 定义

在线蒸馏是指教师模型和学生模型（或多个学生模型）同时训练，互相学习。不需要预训练教师，多个模型在训练过程中互为师生。

#### 3.1.2.2 核心思想

在线蒸馏的核心是：**多个模型同时训练，每个模型都从其他模型的输出中学习。通过互相蒸馏，所有模型的性能都能得到提升。**

最经典的方法：DML（Deep Mutual Learning, 2018）。

#### 3.1.2.3 DML 的数学推导

**（1）DML 的损失函数**

设有 $K$ 个模型 $\{f_1, f_2, \dots, f_K\}$，每个模型 $i$ 的总损失为：

$$
\mathcal{L}_i = \mathcal{L}_{\text{CE}}^{(i)} + \sum_{j \neq i} \mathcal{L}_{\text{KL}}^{(i \leftarrow j)}
$$

其中：

- $\mathcal{L}_{\text{CE}}^{(i)} = \text{CE}(f_i(x), y)$：模型 $i$ 与真实标签的交叉熵损失
- $\mathcal{L}_{\text{KL}}^{(i \leftarrow j)} = \text{KL}\left( \sigma\left( \frac{f_j(x)}{T} \right) \parallel \sigma\left( \frac{f_i(x)}{T} \right) \right)$：模型 $i$ 模仿模型 $j$ 的蒸馏损失

**（2）梯度分析**

对模型 $i$ 的 logit $z_i^{(i)}$ 求梯度：

$$
\frac{\partial \mathcal{L}_i}{\partial z_k^{(i)}} = \frac{\partial \mathcal{L}_{\text{CE}}^{(i)}}{\partial z_k^{(i)}} + \sum_{j \neq i} \frac{\partial \mathcal{L}_{\text{KL}}^{(i \leftarrow j)}}{\partial z_k^{(i)}}
$$

第一项：

$$
\frac{\partial \mathcal{L}_{\text{CE}}^{(i)}}{\partial z_k^{(i)}} = p_k^{(i)} - y_k
$$

第二项（由 2.1.4 节的推导）：

$$
\frac{\partial \mathcal{L}_{\text{KL}}^{(i \leftarrow j)}}{\partial z_k^{(i)}} = \frac{1}{T} \left( q_k^{(i)} - q_k^{(j)} \right)
$$

因此：

$$
\frac{\partial \mathcal{L}_i}{\partial z_k^{(i)}} = (p_k^{(i)} - y_k) + \frac{1}{T} \sum_{j \neq i} \left( q_k^{(i)} - q_k^{(j)} \right)
$$

**（3）DML 的收敛性分析**

**定理**：在 DML 中，每个模型的性能都不会低于其独立训练时的性能。

**直观理解**：每个模型不仅从真实标签学习，还从其他模型的软输出学习。其他模型的输出提供了一种额外的监督信号，相当于一种正则化，有助于提升泛化能力。

**证明思路**：将 DML 的损失重写为：

$$
\mathcal{L}_i = \mathcal{L}_{\text{CE}}^{(i)} + \sum_{j \neq i} \text{KL}^{(i \leftarrow j)}
$$

KL 散度可以分解为：

$$
\text{KL}^{(i \leftarrow j)} = H(q^{(j)}) - H(q^{(j)}, q^{(i)})
$$

其中 $H(q^{(j)})$ 是模型 $j$ 的熵（与模型 $i$ 无关），$H(q^{(j)}, q^{(i)})$ 是交叉熵。因此，最小化 KL 等价于最小化交叉熵，即让模型 $i$ 的输出尽可能匹配模型 $j$ 的输出。

#### 3.1.2.4 DML 的代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def dml_loss(logits_list, labels, T=4.0):
    """
    DML 损失函数
    
    参数:
    - logits_list: 多个模型的 logits 列表, 每个形状 [batch, num_classes]
    - labels: 真实标签, 形状 [batch]
    - T: 温度参数
    
    返回:
    - total_losses: 每个模型的总损失列表
    """
    K = len(logits_list)
    batch_size = labels.size(0)
    total_losses = []
    
    for i in range(K):
        # 交叉熵损失
        ce_loss = F.cross_entropy(logits_list[i], labels)
        
        # 蒸馏损失：模仿其他模型
        kd_loss = 0
        for j in range(K):
            if i != j:
                soft_i = F.log_softmax(logits_list[i] / T, dim=1)
                soft_j = F.softmax(logits_list[j] / T, dim=1)
                kd_loss += F.kl_div(soft_i, soft_j, reduction='batchmean') * (T * T)
        
        kd_loss /= (K - 1)
        total_losses.append(ce_loss + kd_loss)
    
    return total_losses
```

### 3.1.3 自蒸馏（Self-Distillation）

#### 3.1.3.1 定义

自蒸馏是指模型从自身的早期版本或不同分支中学习，不需要额外的教师模型。

#### 3.1.3.2 核心思想

同一模型在不同训练阶段的知识可以互相蒸馏。例如：

- 用训练后期的模型指导训练前期的模型。
- 用深分支指导浅分支。
- 用模型集成指导单个模型。

#### 3.1.3.3 Born-Again Networks（BAN）

**核心思想**：用学生模型重新训练自己。即：

1. 训练一个模型 $f_1$。
2. 用 $f_1$ 作为教师，训练一个相同架构的模型 $f_2$。
3. 用 $f_2$ 作为教师，训练 $f_3$。
4. 重复此过程。

**数学表达**：

$$
\theta_{k+1}^* = \arg\min_{\theta} \ \mathbb{E}_{(x, y)} \left[ \text{KL}\left( \sigma\left( \frac{f_k(x)}{T} \right) \parallel \sigma\left( \frac{f_\theta(x)}{T} \right) \right) + \text{CE}(f_\theta(x), y) \right]
$$

**关键发现**：BAN 的每一代学生模型性能都优于上一代，直到收敛。这说明自蒸馏可以持续提升模型性能。

**理论解释**：自蒸馏相当于一种正则化，每代学生都从教师的软输出中学习到更平滑的决策边界，从而提升泛化能力。

#### 3.1.3.4 基于时间的自蒸馏

**核心思想**：在训练过程中，用后期 epoch 的模型指导前期 epoch 的模型。

**实现**：

```python
class SelfDistillationTrainer:
    def __init__(self, model, T=4.0):
        self.model = model
        self.T = T
        self.teacher_model = None
    
    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            # 每隔几个 epoch 更新教师模型
            if epoch % 5 == 0:
                self.teacher_model = copy.deepcopy(self.model)
                self.teacher_model.eval()
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
            
            for data, labels in train_loader:
                student_logits = self.model(data)
                
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(data)
                    loss = distillation_loss(
                        student_logits, teacher_logits, labels, self.T
                    )
                else:
                    loss = F.cross_entropy(student_logits, labels)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

### 3.1.4 三种蒸馏范式对比

| 维度             | 离线蒸馏   | 在线蒸馏       | 自蒸馏         |
| ---------------- | ---------- | -------------- | -------------- |
| 教师来源         | 预训练模型 | 同步训练的同伴 | 自身的早期版本 |
| 训练阶段         | 两阶段     | 单阶段         | 单阶段         |
| 教师是否更新     | 否         | 是             | 周期性更新     |
| 是否需要额外模型 | 是         | 是（多个学生） | 否             |
| 训练复杂度       | 低         | 高             | 中             |
| 性能上限         | 受教师限制 | 可能超越教师   | 逐步提升       |
| 代表方法         | Hinton KD  | DML            | BAN            |

## 3.2 多教师蒸馏

### 3.2.1 定义

多教师蒸馏（Multi-Teacher Distillation）使用多个教师模型指导学生模型，融合多个教师的知识。

### 3.2.2 核心思想

不同的教师模型可能在不同方面表现优秀。例如：

- 教师 A 在分类任务上准确率高。
- 教师 B 在细粒度识别上表现好。
- 教师 C 对噪声更鲁棒。

通过融合多个教师的知识，学生可以综合多个专家的优势，达到比任何单一教师更好的性能。

### 3.2.3 知识融合策略

#### 3.2.3.1 输出平均

**定义**：对所有教师的软输出取平均。

$$
\mathbf{q}_{\text{ensemble}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{q}_k
$$

其中 $\mathbf{q}_k = \sigma(\mathbf{z}_k / T)$ 是第 $k$ 个教师的软输出。

**优点**：简单、稳定。

**缺点**：所有教师权重相同，无法体现教师之间的差异。

#### 3.2.3.2 加权平均

**定义**：根据教师的重要性赋予不同权重。

$$
\mathbf{q}_{\text{ensemble}} = \sum_{k=1}^{K} w_k \mathbf{q}_k, \quad \sum_{k=1}^{K} w_k = 1, \quad w_k \geq 0
$$

**权重的确定方法**：

**（1）基于验证集性能**

$$
w_k = \frac{\text{Acc}_k}{\sum_{j=1}^{K} \text{Acc}_j}
$$

其中 $\text{Acc}_k$ 是第 $k$ 个教师在验证集上的准确率。

**（2）基于不确定性**

$$
w_k = \frac{1 / \sigma_k^2}{\sum_{j=1}^{K} 1 / \sigma_j^2}
$$

其中 $\sigma_k^2$ 是第 $k$ 个教师预测的不确定性（如熵）。

**（3）基于可学习权重**

将 $w_k$ 作为可训练参数，通过梯度下降优化：

$$
w_k = \frac{\exp(\alpha_k)}{\sum_{j=1}^{K} \exp(\alpha_j)}
$$

其中 $\alpha_k$ 是可学习参数。

#### 3.2.3.3 Logits 平均

**定义**：对所有教师的 logits 取平均。

$$
\mathbf{z}_{\text{ensemble}} = \sum_{k=1}^{K} w_k \mathbf{z}_k
$$

然后对平均 logits 应用 Softmax：

$$
\mathbf{q}_{\text{ensemble}} = \sigma\left( \frac{\mathbf{z}_{\text{ensemble}}}{T} \right)
$$

**与输出平均的区别**：

- 输出平均：先 Softmax 再平均。
- Logits 平均：先平均再 Softmax。

**数学对比**：

输出平均：

$$
\mathbf{q}_{\text{avg}} = \frac{1}{K} \sum_k \frac{\exp(z_k^{(i)} / T)}{\sum_j \exp(z_k^{(j)} / T)}
$$

Logits 平均：

$$
\mathbf{q}_{\text{logit}} = \frac{\exp\left( \frac{1}{K} \sum_k z_k^{(i)} / T \right)}{\sum_j \exp\left( \frac{1}{K} \sum_k z_k^{(j)} / T \right)}
$$

两者通常不同，Logits 平均更接近"共识"，输出平均更保守。

#### 3.2.3.4 特征拼接

**定义**：将多个教师的中间层特征拼接后指导学生。

$$
\mathbf{h}_{\text{ensemble}} = \left[ \mathbf{h}_1; \mathbf{h}_2; \dots; \mathbf{h}_K \right]
$$

学生模仿拼接后的特征：

$$
\mathcal{L}_{\text{feature}} = \left\| \mathbf{h}_S - \phi\left( \left[ \mathbf{h}_1; \mathbf{h}_2; \dots; \mathbf{h}_K \right] \right) \right\|^2
$$

其中 $\phi$ 是维度对齐函数。

### 3.2.4 教师助理（Teacher Assistant）

#### 3.2.4.1 问题背景

当教师和学生之间的能力差距太大时，学生难以直接模仿教师。例如：

- 教师：70B 参数的大模型。
- 学生：1B 参数的小模型。

能力差距太大，蒸馏效果受限。

#### 3.2.4.2 解决方案

引入一个中等规模的"教师助理"（Teacher Assistant, TA）作为桥梁：

```
教师模型（大） → 教师助理（中） → 学生模型（小）
```

**流程**：

1. 用教师模型蒸馏教师助理。
2. 用教师助理蒸馏学生模型。

**数学表达**：

$$
\theta_{\text{TA}}^* = \arg\min_{\theta_{\text{TA}}} \ \text{KL}\left( \sigma\left( \frac{f_T(x)}{T} \right) \parallel \sigma\left( \frac{f_{\text{TA}}(x)}{T} \right) \right)
$$

$$
\theta_S^* = \arg\min_{\theta_S} \ \text{KL}\left( \sigma\left( \frac{f_{\text{TA}}(x)}{T} \right) \parallel \sigma\left( \frac{f_S(x)}{T} \right) \right)
$$

**为什么有效**：

- 教师助理的输出分布比教师的更"温和"，学生更容易模仿。
- 教师助理的容量更接近学生，迁移更平滑。
- 相当于将复杂的知识分解为多个简单的迁移步骤。

### 3.2.5 代码实现

```python
def multi_teacher_distillation_loss(student_logits, teacher_logits_list, 
                                     labels, weights=None, T=4.0, alpha=0.7):
    """
    多教师蒸馏损失函数
    
    参数:
    - student_logits: 学生 logits, 形状 [batch, num_classes]
    - teacher_logits_list: 教师 logits 列表, 每个形状 [batch, num_classes]
    - labels: 真实标签, 形状 [batch]
    - weights: 教师权重, 形状 [K]
    - T: 温度参数
    - alpha: 蒸馏损失权重
    """
    K = len(teacher_logits_list)
    if weights is None:
        weights = [1.0 / K] * K
    
    # 融合教师输出（Logits 平均）
    ensemble_logits = sum(w * tl for w, tl in zip(weights, teacher_logits_list))
    
    # 蒸馏损失
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(ensemble_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    
    # 交叉熵损失
    ce_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * kd_loss + (1 - alpha) * ce_loss
```

## 3.3 跨模态蒸馏

### 3.3.1 定义

跨模态蒸馏（Cross-Modal Distillation）是指教师和学生处理不同模态的数据。例如，教师处理图像，学生处理文本；或者教师处理多模态数据，学生只处理单模态数据。

### 3.3.2 核心思想

不同模态的数据包含互补的信息。通过跨模态蒸馏，可以将一个模态的知识迁移到另一个模态，或者将多模态知识压缩到单模态模型中。

### 3.3.3 典型场景

| 场景          | 教师模态     | 学生模态 | 应用     |
| ------------- | ------------ | -------- | -------- |
| 视觉-语言     | 图像         | 文本     | 文本理解 |
| 语音-文本     | 语音         | 文本     | 文本分类 |
| 3D-2D         | 激光雷达点云 | 相机图像 | 自动驾驶 |
| 多模态-单模态 | 图像+文本    | 文本     | 高效推理 |
| 文本-视觉     | 文本         | 图像     | 图像生成 |

### 3.3.4 数学表达

跨模态蒸馏的损失函数为：

$$
\mathcal{L}_{\text{cross-modal}} = \text{KL}\left( \sigma\left( \frac{f_T(x_T)}{T} \right) \parallel \sigma\left( \frac{f_S(x_S)}{T} \right) \right)
$$

其中：

- $x_T$：教师模态的输入（如图像）
- $x_S$：学生模态的输入（如文本）
- $f_T$：教师模型
- $f_S$：学生模型

**关键挑战**：$x_T$ 和 $x_S$ 是不同模态的数据，需要配对或对齐。

### 3.3.5 对齐方法

#### 3.3.5.1 数据配对

如果有配对数据 $(x_T, x_S)$，可以直接进行蒸馏：

$$
\mathcal{L} = \text{KL}\left( \sigma\left( \frac{f_T(x_T)}{T} \right) \parallel \sigma\left( \frac{f_S(x_S)}{T} \right) \right)
$$

#### 3.3.5.2 特征对齐

如果没有配对数据，可以通过特征对齐建立模态间的映射：

$$
\mathcal{L}_{\text{align}} = \left\| \phi_T(f_T(x_T)) - \phi_S(f_S(x_S)) \right\|^2
$$

其中 $\phi_T$ 和 $\phi_S$ 是将不同模态特征映射到公共空间的函数。

#### 3.3.5.3 对抗对齐

使用对抗训练对齐不同模态的特征分布：

$$
\min_{\phi_T, \phi_S} \max_D \ \mathbb{E}_{x_T}[\log D(\phi_T(f_T(x_T)))] + \mathbb{E}_{x_S}[\log(1 - D(\phi_S(f_S(x_S))))]
$$

### 3.3.6 代码示例

```python
class CrossModalDistillation(nn.Module):
    def __init__(self, teacher_model, student_model, feature_dim=512):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        # 模态对齐层
        self.teacher_proj = nn.Linear(teacher_model.hidden_dim, feature_dim)
        self.student_proj = nn.Linear(student_model.hidden_dim, feature_dim)
    
    def forward(self, image, text, labels):
        # 教师处理图像
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(image)
        
        # 学生处理文本
        student_logits, student_feat = self.student(text)
        
        # 响应蒸馏
        response_loss = F.kl_div(
            F.log_softmax(student_logits / 4.0, dim=1),
            F.softmax(teacher_logits / 4.0, dim=1),
            reduction='batchmean'
        ) * 16.0
        
        # 特征对齐
        teacher_proj = self.teacher_proj(teacher_feat)
        student_proj = self.student_proj(student_feat)
        feature_loss = F.mse_loss(student_proj, teacher_proj.detach())
        
        # 任务损失
        task_loss = F.cross_entropy(student_logits, labels)
        
        return task_loss + 0.5 * response_loss + 0.1 * feature_loss
```

## 3.4 对抗蒸馏

### 3.4.1 定义

对抗蒸馏（Adversarial Distillation）将生成对抗网络（GAN）的思想引入知识蒸馏，通过对抗训练提升蒸馏效果。

### 3.4.2 核心思想

- 学生模型作为生成器，试图生成与教师相似的输出。
- 判别器试图区分学生输出和教师输出。
- 通过对抗训练，学生逐渐学会生成与教师无法区分的输出。

### 3.4.3 数学推导

#### 3.4.3.1 基本对抗蒸馏

**判别器目标**：最大化区分教师和学生的能力。

$$
\max_D \ \mathbb{E}_{x \sim p_{\text{teacher}}}[\log D(x)] + \mathbb{E}_{x \sim p_{\text{student}}}[\log(1 - D(x))]
$$

**学生目标**：最小化判别器的区分能力，即生成与教师相似的输出。

$$
\min_S \ \mathbb{E}_{x \sim p_{\text{student}}}[\log(1 - D(x))]
$$

**联合优化**：

$$
\min_S \max_D \ \mathbb{E}_{x \sim p_{\text{teacher}}}[\log D(x)] + \mathbb{E}_{x \sim p_{\text{student}}}[\log(1 - D(x))]
$$

**理论最优解**：

当学生分布 $p_S$ 等于教师分布 $p_T$ 时，判别器的最优输出为 $D^*(x) = \frac{1}{2}$，此时学生完美模仿教师。

**推导**：对于固定的学生 $p_S$，判别器的最优解为：

$$
D^*(x) = \frac{p_T(x)}{p_T(x) + p_S(x)}
$$

代入目标函数，得到：

$$
\max_D \mathcal{L} = \mathbb{E}_{p_T}\left[ \log \frac{p_T}{p_T + p_S} \right] + \mathbb{E}_{p_S}\left[ \log \frac{p_S}{p_T + p_S} \right]
$$

$$
= -\log 4 + 2 \cdot \text{JSD}(p_T \parallel p_S)
$$

其中 $\text{JSD}$ 是 Jensen-Shannon 散度。因此，最小化对抗损失等价于最小化教师分布和学生分布之间的 JSD。

#### 3.4.3.2 条件对抗蒸馏

在条件对抗蒸馏中，判别器还接收输入 $x$ 作为条件：

$$
\min_S \max_D \ \mathbb{E}_{(x, y) \sim p_{\text{teacher}}}[\log D(x, y)] + \mathbb{E}_{(x, y) \sim p_{\text{student}}}[\log(1 - D(x, y))]
$$

### 3.4.4 优点与缺点

**优点**：

- 可以学习到教师输出分布的精细结构。
- 不依赖于特定的损失函数（如 KL 散度）。
- 在生成任务中效果显著。

**缺点**：

- 训练不稳定，需要精心设计。
- 判别器的设计影响很大。
- 计算开销较大。

### 3.4.5 代码示例

```python
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class AdversarialDistillation:
    def __init__(self, teacher, student, input_dim):
        self.teacher = teacher
        self.student = student
        self.discriminator = Discriminator(input_dim)
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=1e-4)
        self.s_optimizer = Adam(self.student.parameters(), lr=1e-4)
    
    def train_step(self, data, labels):
        # 教师输出
        with torch.no_grad():
            teacher_logits = self.teacher(data)
        
        # 学生输出
        student_logits = self.student(data)
        
        # 训练判别器
        real_labels = torch.ones(data.size(0), 1).cuda()
        fake_labels = torch.zeros(data.size(0), 1).cuda()
        
        d_real = self.discriminator(teacher_logits.detach())
        d_fake = self.discriminator(student_logits.detach())
        
        d_loss = F.binary_cross_entropy(d_real, real_labels) + \
                 F.binary_cross_entropy(d_fake, fake_labels)
        
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # 训练学生
        d_fake_for_student = self.discriminator(student_logits)
        adv_loss = F.binary_cross_entropy(
            d_fake_for_student, real_labels
        )
        task_loss = F.cross_entropy(student_logits, labels)
        s_loss = task_loss + 0.1 * adv_loss
        
        self.s_optimizer.zero_grad()
        s_loss.backward()
        self.s_optimizer.step()
        
        return d_loss.item(), s_loss.item()
```

## 3.5 无数据蒸馏（Data-Free KD）

### 3.5.1 定义

无数据蒸馏（Data-Free Knowledge Distillation）是在没有原始训练数据的情况下进行知识蒸馏。教师模型已训练好，但训练数据不可用（隐私、版权、存储等原因）。

### 3.5.2 核心挑战

- 无法访问教师的训练数据。
- 需要生成合成数据来训练学生。
- 合成数据的质量直接影响蒸馏效果。

### 3.5.3 方法分类

#### 3.5.3.1 基于生成模型的方法

**核心思想**：训练一个生成器，生成与原始数据分布相似的样本。

**流程**：

1. 利用教师的先验知识（如 BatchNorm 统计量）指导生成器。
2. 生成器生成合成样本。
3. 教师对合成样本给出软标签。
4. 学生从教师的软标签中学习。

**数学表达**：

生成器的目标：

$$
\min_G \ \mathcal{L}_{\text{gen}}(G) = \mathcal{L}_{\text{BN}}(G) + \lambda \mathcal{L}_{\text{div}}(G)
$$

其中：

- $\mathcal{L}_{\text{BN}}$：BatchNorm 统计量匹配损失

$$
\mathcal{L}_{\text{BN}} = \sum_{l} \left\| \mu_l(G(z)) - \mu_l^{\text{teacher}} \right\|^2 + \left\| \sigma_l^2(G(z)) - (\sigma_l^{\text{teacher}})^2 \right\|^2
$$

- $\mathcal{L}_{\text{div}}$：多样性损失，防止生成器模式坍塌

$$
\mathcal{L}_{\text{div}} = -\sum_{i \neq j} \left\| G(z_i) - G(z_j) \right\|
$$

学生的蒸馏损失：

$$
\mathcal{L}_{\text{student}} = \mathbb{E}_{z} \left[ \text{KL}\left( \sigma\left( \frac{f_T(G(z))}{T} \right) \parallel \sigma\left( \frac{f_S(G(z))}{T} \right) \right) \right]
$$

#### 3.5.3.2 基于元学习的方法

**核心思想**：学习生成最适合蒸馏的数据。

**数学表达**：

$$
\min_G \ \mathbb{E}_{z} \left[ \mathcal{L}_{\text{distill}}(f_S(G(z)), f_T(G(z))) \right]
$$

通过元学习优化生成器 $G$，使学生模型的蒸馏损失最小。

#### 3.5.3.3 基于教师先验的方法

**核心思想**：直接利用教师的参数（如卷积核、BatchNorm 统计量）作为数据分布的先验。

**方法**：

- 从教师的 BatchNorm 层中提取均值和方差。
- 用这些统计量约束生成数据的分布。
- 使用教师的目标函数作为生成器的正则化。

### 3.5.4 代码示例

```python
class DataFreeDistillation:
    def __init__(self, teacher, student, generator, T=4.0):
        self.teacher = teacher
        self.student = student
        self.generator = generator
        self.T = T
    
    def train_generator(self, num_steps=1000):
        """训练生成器，生成与原始数据分布相似的样本"""
        for step in range(num_steps):
            z = torch.randn(batch_size, latent_dim).cuda()
            fake_data = self.generator(z)
            
            # 匹配教师的 BatchNorm 统计量
            bn_loss = self.match_bn_statistics(fake_data)
            
            # 多样性损失
            diversity_loss = self.compute_diversity_loss(fake_data)
            
            gen_loss = bn_loss + 0.1 * diversity_loss
            
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()
    
    def match_bn_statistics(self, data):
        """匹配教师 BatchNorm 层的统计量"""
        loss = 0
        # 教师前向传播，收集中间统计量
        self.teacher(data)
        teacher_stats = self.teacher.get_bn_stats()
        
        # 学生前向传播
        self.student(data)
        student_stats = self.student.get_bn_stats()
        
        for (mu_t, sigma_t), (mu_s, sigma_s) in zip(teacher_stats, student_stats):
            loss += F.mse_loss(mu_s, mu_t) + F.mse_loss(sigma_s, sigma_t)
        return loss
    
    def train_student(self, num_steps=10000):
        """训练学生模型"""
        for step in range(num_steps):
            z = torch.randn(batch_size, latent_dim).cuda()
            with torch.no_grad():
                fake_data = self.generator(z)
                teacher_logits = self.teacher(fake_data)
            
            student_logits = self.student(fake_data)
            
            loss = F.kl_div(
                F.log_softmax(student_logits / self.T, dim=1),
                F.softmax(teacher_logits / self.T, dim=1),
                reduction='batchmean'
            ) * (self.T * self.T)
            
            self.student_optimizer.zero_grad()
            loss.backward()
            self.student_optimizer.step()
```

## 3.6 图蒸馏（Graph Distillation）

### 3.6.1 定义

图蒸馏（Graph Distillation）是将知识蒸馏应用于图神经网络（GNN），将大容量教师 GNN 的知识迁移到轻量级学生 GNN。

### 3.6.2 核心挑战

- **图结构信息的蒸馏**：不仅仅是节点特征，还有边和邻接关系。
- **邻居聚合模式的蒸馏**：GNN 的核心是邻居聚合，需要蒸馏聚合模式。
- **图规模**：大规模图的蒸馏计算开销大。

### 3.6.3 方法分类

#### 3.6.3.1 基于响应的图蒸馏

学生模仿教师对节点的预测：

$$
\mathcal{L}_{\text{response}} = \sum_{v \in \mathcal{V}} \text{KL}\left( \sigma\left( \frac{\mathbf{z}_v^T}{T} \right) \parallel \sigma\left( \frac{\mathbf{z}_v^S}{T} \right) \right)
$$

其中 $\mathcal{V}$ 是节点集合，$\mathbf{z}_v$ 是节点 $v$ 的 logits。

#### 3.6.3.2 基于特征的图蒸馏

学生模仿教师中间层的节点表示：

$$
\mathcal{L}_{\text{feature}} = \sum_{v \in \mathcal{V}} \left\| \mathbf{h}_v^S - \phi(\mathbf{h}_v^T) \right\|^2
$$

其中 $\mathbf{h}_v$ 是节点 $v$ 的隐藏表示。

#### 3.6.3.3 基于关系的图蒸馏

学生模仿教师中节点之间的关系（如边、邻接矩阵）：

$$
\mathcal{L}_{\text{relation}} = \left\| \mathbf{A}^S - \mathbf{A}^T \right\|_F^2
$$

其中 $\mathbf{A}$ 是邻接矩阵或注意力矩阵。

#### 3.6.3.4 图结构蒸馏

学生模仿教师的图结构（如边的重要性）：

$$
\mathcal{L}_{\text{structure}} = \sum_{(u, v) \in \mathcal{E}} \left\| e_{uv}^S - e_{uv}^T \right\|^2
$$

其中 $e_{uv}$ 是边 $(u, v)$ 的权重或注意力分数。

### 3.6.4 代码示例

```python
class GraphDistillation(nn.Module):
    def __init__(self, teacher_gnn, student_gnn, hidden_dim=256, T=4.0):
        super().__init__()
        self.teacher = teacher_gnn
        self.student = student_gnn
        self.T = T
        self.align = nn.Linear(student_gnn.hidden_dim, teacher_gnn.hidden_dim)
    
    def forward(self, x, edge_index, labels):
        # 教师前向
        with torch.no_grad():
            teacher_logits, teacher_feat = self.teacher(x, edge_index)
        
        # 学生前向
        student_logits, student_feat = self.student(x, edge_index)
        
        # 响应蒸馏
        response_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T * self.T)
        
        # 特征蒸馏
        student_feat_aligned = self.align(student_feat)
        feature_loss = F.mse_loss(student_feat_aligned, teacher_feat.detach())
        
        # 任务损失
        task_loss = F.cross_entropy(student_logits, labels)
        
        return task_loss + 0.5 * response_loss + 0.1 * feature_loss
```

## 3.7 蒸馏技术对比

### 3.7.1 全面对比表

| 技术       | 核心思想       | 优点           | 缺点         | 适用场景     | 代表方法    |
| ---------- | -------------- | -------------- | ------------ | ------------ | ----------- |
| 离线蒸馏   | 预训练教师指导 | 稳定、简单     | 两阶段训练   | 通用         | Hinton KD   |
| 在线蒸馏   | 多模型互相学习 | 无需预训练教师 | 训练复杂     | 无预训练教师 | DML         |
| 自蒸馏     | 模型自我改进   | 无需额外模型   | 提升有限     | 泛化提升     | BAN         |
| 多教师蒸馏 | 融合多个教师   | 知识丰富       | 计算开销大   | 集成学习     | Ensemble KD |
| 跨模态蒸馏 | 不同模态间蒸馏 | 模态迁移       | 对齐困难     | 多模态       | CMKD        |
| 对抗蒸馏   | 对抗训练       | 效果好         | 训练不稳定   | 高质量蒸馏   | AD          |
| 无数据蒸馏 | 无原始数据     | 隐私保护       | 依赖合成质量 | 隐私场景     | DFKD        |
| 图蒸馏     | GNN 蒸馏       | 图结构迁移     | 计算复杂     | 图数据       | GKD         |

### 3.7.2 技术选择决策树

```
是否需要隐私保护？
├── 是 → 无数据蒸馏
└── 否 → 是否有预训练教师？
    ├── 是 → 离线蒸馏
    │   ├── 教师数量 > 1？→ 多教师蒸馏
    │   └── 教师数量 = 1？
    │       ├── 模态不同？→ 跨模态蒸馏
    │       └── 模态相同？
    │           ├── 需要高质量？→ 对抗蒸馏
    │           └── 一般需求 → 标准离线蒸馏
    └── 否 → 在线蒸馏 / 自蒸馏
        ├── 有多个模型？→ 在线蒸馏（DML）
        └── 只有单个模型？→ 自蒸馏（BAN）
```

### 3.7.3 组合使用策略

多种蒸馏技术可以组合使用：

```
原始数据 → 离线蒸馏（基础）→ 对抗蒸馏（提升质量）→ 多教师蒸馏（融合知识）→ 部署
```

**组合示例**：

| 组合            | 效果              | 适用场景   |
| --------------- | ----------------- | ---------- |
| 离线 + 多教师   | 融合多专家知识    | 集成学习   |
| 离线 + 对抗     | 高质量蒸馏        | 生成任务   |
| 在线 + 自蒸馏   | 持续提升          | 半监督学习 |
| 无数据 + 对抗   | 隐私保护 + 高质量 | 隐私场景   |
| 跨模态 + 多教师 | 多模态知识融合    | 多模态学习 |



# 四、模型剪枝

## 4.1 什么是模型剪枝

### 4.1.1 定义

模型剪枝（Model Pruning）是一种模型压缩技术，通过移除神经网络中冗余的参数、神经元、通道或层，减少模型的参数量和计算量，同时尽可能保持模型性能。

**形式化定义**：

设原始模型参数为 $\theta \in \mathbb{R}^P$，剪枝的目标是找到一个稀疏的参数 $\theta' \in \mathbb{R}^P$，使得 $\|\theta'\|_0 \ll \|\theta\|_0$，同时满足：

$$
\min_{\theta'} \ \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \mathcal{L}(f_{\theta'}(x), y) \right] \quad \text{s.t.} \quad \frac{\|\theta'\|_0}{\|\theta\|_0} \leq 1 - s
$$

其中：

- $\|\cdot\|_0$：L0 范数，表示非零元素个数
- $s$：目标稀疏率（如 $s = 0.9$ 表示保留 10% 的参数）
- $\mathcal{D}$：数据分布
- $\mathcal{L}$：任务损失函数

**简化的剪枝目标**：

$$
\min_{\mathbf{M}, \theta} \ \mathcal{L}(f_{\theta \odot \mathbf{M}}(x), y) \quad \text{s.t.} \quad \|\mathbf{M}\|_0 \leq K
$$

其中 $\mathbf{M} \in \{0, 1\}^P$ 是二值掩码矩阵，$\odot$ 是逐元素乘法，$K$ 是保留的参数数量。

### 4.1.2 核心思想

深度神经网络通常存在大量冗余参数。Denil 等人（2013）的研究表明，深度网络的参数数量远超实际需要的数量，大部分参数对最终预测的贡献很小。剪枝的核心思想是：**识别并移除这些冗余参数，得到一个更紧凑的模型，同时保持性能。**

**关键观察**：

- 训练后，权重矩阵中大量参数接近零。
- 移除这些小权重对模型输出的影响很小。
- 剪枝后的模型可以通过微调恢复性能。

**冗余性的数学解释**：

神经网络的参数矩阵通常是低秩的或近似低秩的。设权重矩阵 $\mathbf{W} \in \mathbb{R}^{d \times k}$，其奇异值分解为：

$$
\mathbf{W} = \sum_{i=1}^{\min(d, k)} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

其中 $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$ 是奇异值。实验发现，前 $r$ 个奇异值（$r \ll \min(d, k)$）已经捕获了 $\mathbf{W}$ 的绝大部分能量：

$$
\frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{\min(d, k)} \sigma_i^2} \geq 1 - \epsilon
$$

这说明权重矩阵的**有效秩**远小于其名义秩，存在大量冗余。

### 4.1.3 剪枝与知识蒸馏的区别

|   维度   |     知识蒸馏     |               模型剪枝               |
| :------: | :--------------: | :----------------------------------: |
|   原理   | 师生框架迁移知识 |             移除冗余参数             |
| 输出模型 |  可以是全新架构  |        通常保持原架构（稀疏）        |
|  参数量  | 减少（结构更小） |           减少（部分为零）           |
| 推理加速 |  直接（小模型）  |           需要稀疏计算支持           |
| 训练方式 |   需要教师指导   |             通常需要微调             |
|  灵活性  | 高（架构可不同） |           低（保持原架构）           |
| 硬件支持 |     通用硬件     | 结构化剪枝通用，非结构化需要专用硬件 |

### 4.1.4 剪枝的历史背景

**（1）早期工作（1989-1990）**

- **OBD（Optimal Brain Damage, 1989）**：LeCun 等人提出基于 Hessian 矩阵的权重重要性评估，移除不重要的权重。
- **OBS（Optimal Brain Surgeon, 1990）**：Hassibi 和 Stork 改进了 OBD，考虑权重之间的相关性。

**（2）深度学习时代的剪枝（2015至今）**

- **Han 等人（2015）**：提出"Deep Compression"流程：剪枝 → 量化 → 霍夫曼编码，将 AlexNet 压缩 35 倍。
- **结构化剪枝（2016-2018）**：Li 等人提出基于 L1 范数的通道剪枝，Liu 等人提出基于 BN 缩放因子的通道剪枝。
- **彩票假设（2019）**：Frankle 和 Carbin 提出 Lottery Ticket Hypothesis，引发了对剪枝本质的深入思考。
- **LLM 剪枝（2023至今）**：SparseGPT、Wanda 等方法将剪枝应用于大语言模型。

## 4.2 剪枝的分类

### 4.2.1 按剪枝对象分类

#### 4.2.1.1 权重剪枝（Weight Pruning）

移除个别权重参数（将其置零），不改变网络结构。

**数学表达**：

$$
\mathbf{W}' = \mathbf{W} \odot \mathbf{M}, \quad M_{ij} \in \{0, 1\}
$$

**特点**：

- 粒度最细，灵活性最高。
- 可以实现很高的稀疏率。
- 但需要专用硬件支持稀疏计算。

#### 4.2.1.2 神经元剪枝（Neuron Pruning）

移除整个神经元（及其所有输入输出连接），改变网络结构。

**数学表达**：

对于第 $l$ 层的第 $j$ 个神经元，移除后：

$$
\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)}[\mathcal{S}, :], \quad \mathbf{W}^{(l+1)} \leftarrow \mathbf{W}^{(l+1)}[:, \mathcal{S}]
$$

其中 $\mathcal{S}$ 是保留的神经元索引集合。

**特点**：

- 剪枝后的模型是标准稠密模型。
- 可以直接在通用硬件上加速。
- 剪枝率通常较低。

#### 4.2.1.3 通道剪枝（Channel Pruning）

移除整个通道（卷积核的某一维），适用于 CNN。

**数学表达**：

对于第 $l$ 层的权重 $\mathbf{W}^{(l)} \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$，移除 $C_{pruned}$ 个输出通道：

$$
\mathbf{W}'^{(l)} = \mathbf{W}^{(l)}[\mathcal{S}, :, :, :], \quad |\mathcal{S}| = C_{out} - C_{pruned}
$$

同时需要移除下一层对应的输入通道：

$$
\mathbf{W}'^{(l+1)} = \mathbf{W}^{(l+1)}[:, \mathcal{S}, :, :]
$$

**特点**：

- 适用于 CNN 和 Transformer。
- 可以直接在通用硬件上加速。
- 是结构化剪枝的主要形式。

#### 4.2.1.4 层剪枝（Layer Pruning）

移除整个层，适用于深层网络。

**数学表达**：

移除第 $l$ 层后，第 $l-1$ 层的输出直接连接到第 $l+1$ 层的输入。

**特点**：

- 剪枝率最高（直接减少深度）。
- 但可能导致较大的性能下降。
- 通常用于非常深的网络（如 ResNet-152）。

### 4.2.2 按剪枝时机分类

#### 4.2.2.1 训练后剪枝（Post-training Pruning）

先训练完整模型，再剪枝，最后微调。

**流程**：

```
步骤1：训练完整模型
步骤2：剪枝（移除冗余参数）
步骤3：微调（恢复性能）
```

**优点**：简单、通用。

**缺点**：需要完整的训练和微调流程。

#### 4.2.2.2 训练中剪枝（Pruning during Training）

在训练过程中逐步剪枝，如逐步增加稀疏性约束。

**流程**：

```
步骤1：开始训练
步骤2：每隔一定步数，剪枝一部分参数
步骤3：继续训练
步骤4：重复步骤2-3，直到达到目标稀疏率
```

**优点**：模型有时间适应稀疏结构，性能损失更小。

**缺点**：训练流程复杂。

#### 4.2.2.3 训练前剪枝（Pruning before Training）

先确定稀疏结构，再从头训练。

**流程**：

```
步骤1：随机初始化模型
步骤2：剪枝（随机或基于初始化）
步骤3：训练剪枝后的模型
```

**代表方法**：彩票假设（Lottery Ticket Hypothesis）。

**优点**：训练成本低（模型更小）。

**缺点**：需要找到好的稀疏结构。

### 4.2.3 按剪枝粒度分类

#### 4.2.3.1 非结构化剪枝（Unstructured Pruning）

移除个别权重，保持网络结构不变。

**数学表达**：

$$
\mathbf{W}' = \mathbf{W} \odot \mathbf{M}, \quad M_{ij} \in \{0, 1\}
$$

**优点**：

- 可以实现很高的稀疏率（90% 以上）。
- 对性能的影响较小。

**缺点**：

- 稀疏矩阵的推理需要专用硬件或库支持。
- 通用硬件上难以获得实际的加速。

#### 4.2.3.2 结构化剪枝（Structured Pruning）

移除整个结构单元（如通道、神经元、层），剪枝后的网络形状改变。

**数学表达**：

对于通道剪枝：

$$
\mathbf{W}'^{(l)} = \mathbf{W}^{(l)}[\mathcal{S}, :, :, :]
$$

**优点**：

- 剪枝后的模型是标准的稠密模型，可以直接在通用硬件上加速。
- 不需要专用硬件支持。

**缺点**：

- 剪枝率通常低于非结构化剪枝。
- 对性能的影响可能更大。

#### 4.2.3.3 半结构化剪枝

结合结构化和非结构化剪枝，如 N:M 稀疏（每 M 个连续权重中保留 N 个非零）。

**数学表达**：

对于 N:M 稀疏，每个大小为 M 的块中保留绝对值最大的 N 个权重：

$$
\mathbf{W}'_{ij} = \begin{cases} W_{ij} & \text{if } |W_{ij}| \in \text{Top-N of block} \\ 0 & \text{otherwise} \end{cases}
$$

**NVIDIA Ampere 支持**：2:4 稀疏（每 4 个权重中保留 2 个），可获得 2 倍加速。

### 4.2.4 剪枝分类总结表

| 分类维度 | 类型       | 粒度     | 加速方式 | 典型稀疏率 |
| -------- | ---------- | -------- | -------- | ---------- |
| 剪枝对象 | 权重剪枝   | 单个权重 | 稀疏计算 | 90%+       |
| 剪枝对象 | 神经元剪枝 | 神经元   | 稠密计算 | 50%-80%    |
| 剪枝对象 | 通道剪枝   | 通道     | 稠密计算 | 50%-70%    |
| 剪枝对象 | 层剪枝     | 层       | 稠密计算 | 30%-50%    |
| 剪枝时机 | 训练后剪枝 | —        | —        | —          |
| 剪枝时机 | 训练中剪枝 | —        | —        | —          |
| 剪枝时机 | 训练前剪枝 | —        | —        | —          |
| 剪枝粒度 | 非结构化   | 单个权重 | 稀疏计算 | 90%+       |
| 剪枝粒度 | 结构化     | 通道/层  | 稠密计算 | 50%-70%    |
| 剪枝粒度 | 半结构化   | N:M 块   | 硬件加速 | 50%        |

## 4.3 结构化剪枝

### 4.3.1 定义

结构化剪枝移除整个结构单元（如通道、神经元、层），剪枝后的网络是标准的稠密网络。

### 4.3.2 核心方法

#### 4.3.2.1 基于 L1 范数的通道剪枝

**核心思想**：计算每个通道的 L1 范数，移除范数最小的通道。

**数学表达**：

对于第 $l$ 层的第 $c$ 个输出通道，其 L1 范数为：

$$
\text{Score}_c = \sum_{k=1}^{K} \sum_{k'=1}^{K} \sum_{i=1}^{C_{in}} |W_{c,i,k,k'}^{(l)}|
$$

移除得分最低的 $C_{pruned}$ 个通道：

$$
\mathcal{S} = \text{Top-}(C_{out} - C_{pruned}) \text{ of } \{\text{Score}_c\}_{c=1}^{C_{out}}
$$

**推导依据**：

L1 范数衡量通道的"重要性"。L1 范数越小，说明该通道的权重越接近零，对输出的贡献越小。

**代码实现**：

```python
def l1_channel_pruning(model, pruning_rate=0.5):
    """基于 L1 范数的通道剪枝"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 计算每个输出通道的 L1 范数
            weight = module.weight.data
            l1_norm = weight.abs().sum(dim=(1, 2, 3))
            
            # 确定阈值
            num_channels = weight.size(0)
            num_prune = int(num_channels * pruning_rate)
            threshold = torch.topk(l1_norm, num_prune, largest=False).values[-1]
            
            # 生成掩码
            mask = (l1_norm > threshold).float().view(-1, 1, 1, 1)
            module.weight.data *= mask
    
    return model
```

#### 4.3.2.2 基于 BN 缩放因子的通道剪枝

**核心思想**：利用 BatchNorm 层的缩放因子 $\gamma$ 作为通道重要性的度量。

**数学表达**：

BN 层的输出为：

$$
\hat{x}_c = \gamma_c \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c
$$

$\gamma_c$ 越小，说明该通道的重要性越低。移除 $\gamma_c$ 小于阈值的通道：

$$
\mathcal{S} = \{c : \gamma_c > \tau\}
$$

**训练时的稀疏化约束**：

在损失函数中添加 $\gamma$ 的 L1 正则化：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{c} |\gamma_c|
$$

这促使网络自动学习哪些通道重要，哪些可以剪枝。

**推导**：

L1 正则化对 $\gamma_c$ 的次梯度为：

$$
\frac{\partial \mathcal{L}}{\partial \gamma_c} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial \gamma_c} + \lambda \cdot \text{sign}(\gamma_c)
$$

当 $\gamma_c > 0$ 时，梯度中包含 $+\lambda$，推动 $\gamma_c$ 减小。当 $\gamma_c < 0$ 时，梯度中包含 $-\lambda$，推动 $\gamma_c$ 增大。因此，L1 正则化促使 $\gamma_c$ 趋向于零。

**代码实现**：

```python
class PrunableBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
    
    def forward(self, x):
        # 标准 BN 前向
        return super().forward(x)
    
    def get_gamma(self):
        return self.weight.data

def bn_sparsity_loss(model, lambda_sparse=1e-4):
    """BN 缩放因子的 L1 正则化损失"""
    loss = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss += module.weight.abs().sum()
    return lambda_sparse * loss
```

#### 4.3.2.3 基于泰勒展开的通道剪枝

**核心思想**：利用泰勒展开估计移除某个通道对损失的影响。

**数学推导**：

设损失函数为 $\mathcal{L}$，移除第 $c$ 个通道后的损失变化可以用一阶泰勒展开近似：

$$
\Delta \mathcal{L}_c = \mathcal{L}(\mathbf{W}' = \mathbf{W} \text{ with channel } c \text{ pruned}) - \mathcal{L}(\mathbf{W})
$$

对于通道 $c$，其输出为 $z_c$，移除后 $z_c = 0$。损失的变化为：

$$
\Delta \mathcal{L}_c = \mathcal{L}(z_c = 0) - \mathcal{L}(z_c)
$$

一阶泰勒展开：

$$
\mathcal{L}(z_c = 0) \approx \mathcal{L}(z_c) - \frac{\partial \mathcal{L}}{\partial z_c} \cdot z_c
$$

因此：

$$
\Delta \mathcal{L}_c \approx -\frac{\partial \mathcal{L}}{\partial z_c} \cdot z_c
$$

移除对损失影响最小的通道：

$$
\mathcal{S} = \text{Top-}(C_{out} - C_{pruned}) \text{ of } \left\{ \left| \frac{\partial \mathcal{L}}{\partial z_c} \cdot z_c \right| \right\}
$$

**代码实现**：

```python
def taylor_channel_pruning(model, dataloader, pruning_rate=0.5):
    """基于泰勒展开的通道剪枝"""
    # 收集每个通道的梯度 × 激活值
    scores = {}
    
    def hook_fn(module, grad_input, grad_output):
        # grad_output: 损失对通道输出的梯度
        # output: 通道的输出值
        name = module._name
        if name not in scores:
            scores[name] = 0
        scores[name] += (grad_output[0] * module.output).abs().sum(dim=(0, 2, 3))
    
    # 注册 hook
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module._name = name
            module.register_forward_hook(
                lambda m, i, o: setattr(m, 'output', o)
            )
            module.register_full_backward_hook(hook_fn)
    
    # 运行校准数据
    for data, _ in dataloader:
        output = model(data)
        loss = output.sum()
        loss.backward()
    
    # 剪枝
    # ...
    return model
```

### 4.3.3 优缺点

**优点**：

- 剪枝后的模型是标准的稠密模型，可直接加速。
- 不需要专用硬件支持。
- 实现相对简单。

**缺点**：

- 剪枝率通常低于非结构化剪枝（通常 2-4×）。
- 剪枝可能导致较大的性能下降，需要微调恢复。
- 对于某些层（如第一层和最后一层），剪枝可能严重影响性能。

## 4.4 非结构化剪枝

### 4.4.1 定义

非结构化剪枝移除个别权重参数，剪枝后的权重矩阵是稀疏的，但形状不变。

### 4.4.2 核心方法

#### 4.4.2.1 基于幅度的剪枝

**核心思想**：移除绝对值最小的权重。

**数学表达**：

$$
\mathbf{M}_{ij} = \begin{cases} 1 & \text{if } |W_{ij}| > \tau \\ 0 & \text{if } |W_{ij}| \leq \tau \end{cases}
$$

其中 $\tau$ 是阈值，通常根据目标稀疏率确定。

**全局剪枝 vs 逐层剪枝**：

- **全局剪枝**：在所有层的所有权重中统一选择阈值 $\tau$。
- **逐层剪枝**：每层独立选择阈值，使得每层的稀疏率相同。

**全局剪枝的阈值确定**：

设目标稀疏率为 $s$，则阈值 $\tau$ 是所有权重绝对值的第 $s$ 百分位数：

$$
\tau = \text{Percentile}\left( \{|W_{ij}|\}, s \right)
$$

**推导**：剪枝后的非零权重数量为 $(1-s) \cdot P$，其中 $P$ 是总参数量。因此，$\tau$ 应该使得 $|\{ (i,j) : |W_{ij}| > \tau \}| = (1-s) \cdot P$。这就是第 $s$ 百分位数。

**逐层剪枝的阈值确定**：

对于第 $l$ 层，设该层参数量为 $P_l$，目标稀疏率为 $s$，则阈值 $\tau_l$ 是该层权重绝对值的第 $s$ 百分位数。

**代码实现**：

```python
def magnitude_pruning(model, sparsity=0.9, global_pruning=True):
    """
    基于幅度的剪枝
    
    参数:
    - model: 待剪枝的模型
    - sparsity: 目标稀疏率
    - global_pruning: 是否全局剪枝
    """
    if global_pruning:
        # 全局剪枝：所有层的权重统一选择阈值
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.abs().flatten())
        all_weights = torch.cat(all_weights)
        
        # 确定阈值
        threshold = torch.quantile(all_weights, sparsity)
        
        # 应用掩码
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = (param.data.abs() > threshold).float()
                masks[name] = mask
                param.data *= mask
    else:
        # 逐层剪枝：每层独立选择阈值
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                threshold = torch.quantile(param.data.abs().flatten(), sparsity)
                mask = (param.data.abs() > threshold).float()
                masks[name] = mask
                param.data *= mask
    
    return model, masks
```

#### 4.4.2.2 基于重要性的剪枝

**核心思想**：根据权重对损失的重要性进行剪枝。

**重要性度量**：

**（1）基于梯度**

$$
I_{ij} = \left| W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}} \right|
$$

**推导**：损失对权重的一阶泰勒展开：

$$
\mathcal{L}(W_{ij} = 0) \approx \mathcal{L}(W_{ij}) - W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}}
$$

因此，$\left| W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}} \right|$ 是移除权重 $W_{ij}$ 后损失变化的近似。重要性越小，移除后损失变化越小。

**（2）基于 Hessian**

$$
I_{ij} = W_{ij}^2 \cdot H_{ij,ij}
$$

其中 $H_{ij,ij}$ 是 Hessian 矩阵的对角元素。

**推导**：损失对权重的二阶泰勒展开：

$$
\mathcal{L}(W_{ij} = 0) \approx \mathcal{L}(W_{ij}) - W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}} + \frac{1}{2} W_{ij}^2 \cdot H_{ij,ij}
$$

在最优解处 $\frac{\partial \mathcal{L}}{\partial W_{ij}} \approx 0$，因此：

$$
\mathcal{L}(W_{ij} = 0) - \mathcal{L}(W_{ij}) \approx \frac{1}{2} W_{ij}^2 \cdot H_{ij,ij}
$$

**（3）基于 Fisher 信息**

$$
I_{ij} = W_{ij}^2 \cdot F_{ij}
$$

其中 $F_{ij}$ 是 Fisher 信息矩阵的对角元素：

$$
F_{ij} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \left( \frac{\partial \log p(y \mid x, \theta)}{\partial W_{ij}} \right)^2 \right]
$$

#### 4.4.2.3 迭代剪枝

**核心思想**：不要一次性剪枝到目标稀疏率，而是逐步剪枝，每次剪枝后微调。

**流程**：

```
步骤1：训练完整模型
步骤2：剪枝 10%-20% 的权重
步骤3：微调恢复性能
步骤4：重复步骤2-3，直到达到目标稀疏率
```

**为什么迭代剪枝比一次性剪枝好？**

一次性剪枝到 90% 稀疏率可能导致性能严重下降。迭代剪枝每次只剪枝少量权重，模型有时间通过微调适应新的稀疏结构。

**数学分析**：

设一次剪枝的稀疏率为 $s_1$，剪枝后的性能损失为 $\Delta \mathcal{L}_1$。迭代剪枝 $k$ 次，每次稀疏率为 $s_1/k$，则总性能损失为：

$$
\Delta \mathcal{L}_{\text{iterative}} = \sum_{i=1}^{k} \Delta \mathcal{L}_i
$$

由于每次剪枝量小，$\Delta \mathcal{L}_i$ 较小，且微调后可以恢复。因此：

$$
\Delta \mathcal{L}_{\text{iterative}} < \Delta \mathcal{L}_{\text{one-shot}}
$$

**代码实现**：

```python
def iterative_pruning(model, train_loader, target_sparsity=0.9,
                      num_rounds=10, finetune_epochs=1):
    """
    迭代剪枝
    
    参数:
    - model: 待剪枝的模型
    - train_loader: 训练数据
    - target_sparsity: 目标稀疏率
    - num_rounds: 剪枝轮数
    - finetune_epochs: 每轮微调的 epoch 数
    """
    current_sparsity = 0
    sparsity_per_round = target_sparsity / num_rounds
    
    for round in range(num_rounds):
        # 剪枝
        current_sparsity += sparsity_per_round
        model, masks = magnitude_pruning(model, current_sparsity)
        
        # 微调
        finetune(model, train_loader, epochs=finetune_epochs)
        
        print(f"Round {round+1}: Sparsity = {current_sparsity:.2%}")
    
    return model, masks
```

### 4.4.3 优缺点

**优点**：

- 可以实现很高的稀疏率（90%-99%）。
- 对性能的影响较小。
- 实现简单。

**缺点**：

- 稀疏矩阵需要专用硬件或库（如 cuSPARSE）支持。
- 在通用硬件上难以获得实际加速。
- 稀疏矩阵的存储和计算效率低于稠密矩阵。

## 4.5 剪枝算法

### 4.5.1 基于幅度的剪枝

**算法**：

```
输入：训练好的模型 W，目标稀疏率 s
输出：剪枝后的模型 W'

1. 计算所有参数的绝对值
2. 确定阈值 τ = Percentile({|W_ij|}, s)
3. 生成掩码 M_ij = 1 if |W_ij| > τ else 0
4. 应用掩码：W' = W ⊙ M
5. 微调 W' 恢复性能
```

### 4.5.2 基于重要性的剪枝

**算法**：

```
输入：训练好的模型 W，目标稀疏率 s
输出：剪枝后的模型 W'

1. 对每个权重计算重要性得分 I_ij
2. 确定阈值 τ = Percentile({I_ij}, s)
3. 生成掩码 M_ij = 1 if I_ij > τ else 0
4. 应用掩码：W' = W ⊙ M
5. 微调 W' 恢复性能
```

### 4.5.3 基于梯度的剪枝

**核心思想**：利用梯度信息估计权重的重要性。

**数学表达**：

对于权重 $W_{ij}$，其重要性为：

$$
I_{ij} = \left| W_{ij} \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}} \right|
$$

### 4.5.4 基于正则化的剪枝

**核心思想**：在训练时添加稀疏性正则化，使模型自动学习稀疏权重。

**L1 正则化**：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{ij} |W_{ij}|
$$

L1 正则化促使权重趋向于零，训练后可以直接剪枝小权重。

**L0 正则化**：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{ij} \mathbb{1}[W_{ij} \neq 0]
$$

L0 正则化直接惩罚非零参数个数，但不可微，需要使用近似方法（如 Hard Concrete 分布）。

**组 Lasso**：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_{g} \| \mathbf{W}_g \|_2
$$

其中 $\mathbf{W}_g$ 是第 $g$ 组的权重（如一个通道）。组 Lasso 促使整组权重为零，实现结构化剪枝。

### 4.5.5 彩票假设（Lottery Ticket Hypothesis）

#### 4.5.5.1 核心思想

Frankle 和 Carbin（2019）提出彩票假设：**一个随机初始化的密集神经网络包含一个子网络（"中奖彩票"），当独立训练时，该子网络可以在相似的迭代次数内达到与原始网络相当的测试精度。**

#### 4.5.5.2 数学表述

设原始网络为 $f(x; \theta_0)$，其中 $\theta_0$ 是初始参数。存在一个掩码 $\mathbf{M}$，使得子网络 $f(x; \theta_0 \odot \mathbf{M})$ 在训练后性能与原始网络相当：

$$
\min_{\mathbf{M}} \ \left| \text{Acc}(f(x; \theta_0 \odot \mathbf{M})) - \text{Acc}(f(x; \theta_0)) \right|
$$

**迭代幅度剪枝（IMP）算法**：

```
1. 随机初始化网络，保存初始权重 θ_0
2. 训练网络，得到权重 θ_T
3. 剪枝 |θ_T| 最小的 p% 权重，得到掩码 M
4. 将权重重置为初始值：θ = θ_0 ⊙ M
5. 重复步骤2-4，直到达到目标稀疏率
```

#### 4.5.5.3 关键发现

- 中奖彩票确实存在，且可以在多个数据集和架构上找到。
- 中奖彩票的初始化至关重要，随机重新初始化后无法恢复性能。
- 中奖彩票的稀疏率可以达到 90% 以上。

#### 4.5.5.4 代码实现

```python
def lottery_ticket_training(model, train_loader, epochs, pruning_rate=0.2,
                            num_rounds=5):
    """彩票假设的迭代幅度剪枝"""
    # 保存初始权重
    initial_weights = {name: param.clone() 
                       for name, param in model.named_parameters()}
    
    masks = {name: torch.ones_like(param)
             for name, param in model.named_parameters()}
    
    for round in range(num_rounds):
        # 训练
        train(model, train_loader, epochs)
        
        # 剪枝
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append((param.data * masks[name]).abs().flatten())
        all_weights = torch.cat(all_weights)
        
        threshold = torch.quantile(all_weights, pruning_rate)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                new_mask = (param.data.abs() > threshold).float()
                masks[name] *= new_mask
        
        # 重置为初始权重
        for name, param in model.named_parameters():
            param.data = initial_weights[name] * masks[name]
    
    return model, masks
```

## 4.6 剪枝与蒸馏的结合

### 4.6.1 先剪枝后蒸馏

**流程**：

```
步骤1：训练大模型（教师）
步骤2：剪枝大模型得到剪枝后的模型
步骤3：用剪枝后的模型作为教师，蒸馏到学生模型
```

**优点**：剪枝后的教师模型更紧凑，蒸馏效率更高。

### 4.6.2 先蒸馏后剪枝

**流程**：

```
步骤1：训练大模型（教师）
步骤2：蒸馏得到学生模型
步骤3：剪枝学生模型
步骤4：微调剪枝后的学生模型
```

**优点**：学生模型已经较小，剪枝的计算开销更低。

### 4.6.3 联合优化

**流程**：

```
步骤1：同时进行蒸馏和剪枝
步骤2：在损失函数中同时考虑蒸馏损失和稀疏性约束
```

**数学表达**：

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{KD}} + (1 - \alpha) \mathcal{L}_{\text{CE}} + \lambda \mathcal{R}_{\text{sparsity}}
$$

其中 $\mathcal{R}_{\text{sparsity}}$ 是稀疏性正则化项。

### 4.6.4 剪枝与蒸馏的协同效应

**（1）剪枝提升蒸馏效果**

- 剪枝后的教师模型更紧凑，其输出分布更"温和"，学生更容易模仿。
- 剪枝去除了教师模型中的噪声，使知识更纯净。

**（2）蒸馏提升剪枝效果**

- 蒸馏损失可以作为剪枝后的微调目标，比单纯的交叉熵更有效。
- 蒸馏可以帮助剪枝后的模型恢复性能。

**（3）实验验证**

| 方法        | 准确率 | 参数量 | 说明          |
| ----------- | ------ | ------ | ------------- |
| 原始模型    | 95.0%  | 100%   | 基线          |
| 仅剪枝      | 92.5%  | 30%    | 性能下降 2.5% |
| 仅蒸馏      | 94.0%  | 50%    | 性能下降 1.0% |
| 剪枝 + 蒸馏 | 94.5%  | 30%    | 性能下降 0.5% |
| 蒸馏 + 剪枝 | 94.3%  | 30%    | 性能下降 0.7% |

# 五、模型量化

## 5.1 什么是模型量化

### 5.1.1 定义

模型量化（Model Quantization）是一种模型压缩技术，通过将模型参数从高精度浮点数（如 FP32）转换为低精度整数（如 INT8、INT4），减少模型的存储和计算开销。

**形式化定义**：

设原始权重为 $W \in \mathbb{R}^{d \times k}$，量化的目标是将 $W$ 映射为低精度表示 $\hat{W}$：

$$
\hat{W} = Q(W)
$$

其中 $Q$ 是量化函数。量化后的模型推理时使用低精度计算，输出与原始模型尽可能接近。

### 5.1.2 核心思想

深度神经网络的权重和激活值对精度的要求并不高。研究表明，将 FP32 量化为 INT8 后，模型的性能损失通常很小（<1%）。量化的核心思想是：**用更少的比特表示模型的参数和计算，减少存储和计算开销，同时保持性能。**

### 5.1.3 量化与蒸馏、剪枝的关系

| 维度     | 知识蒸馏           | 剪枝               | 量化               |
| -------- | ------------------ | ------------------ | ------------------ |
| 压缩对象 | 模型知识           | 模型参数           | 参数精度           |
| 输出模型 | 小模型             | 稀疏模型           | 低精度模型         |
| 压缩率   | 2-10×              | 2-10×              | 4×（INT8）         |
| 硬件支持 | 通用               | 稀疏支持           | 量化支持           |
| 性能损失 | 小                 | 中等               | 中等               |
| 组合使用 | 可与剪枝、量化结合 | 可与蒸馏、量化结合 | 可与蒸馏、剪枝结合 |

## 5.2 量化的数学基础

### 5.2.1 量化函数

#### 5.2.1.1 均匀量化

**定义**：将浮点数均匀映射到整数。

**数学表达**：

$$
q = \text{round}\left( \frac{W}{s} \right) + z
$$

其中：

- $s$：缩放因子（scale）
- $z$：零点（zero point）
- $q$：量化后的整数

**反量化**：

$$
\hat{W} = s \cdot (q - z)
$$

**缩放因子和零点的确定**：

$$
s = \frac{W_{\max} - W_{\min}}{2^b - 1}
$$

$$
z = \text{round}\left( \frac{-W_{\min}}{s} \right)
$$

其中 $b$ 是量化位数。

**推导**：量化范围 $[W_{\min}, W_{\max}]$ 映射到整数范围 $[0, 2^b - 1]$。

$$
q = \frac{W - W_{\min}}{s} = \frac{W - W_{\min}}{(W_{\max} - W_{\min}) / (2^b - 1)}
$$

$$
= (2^b - 1) \frac{W - W_{\min}}{W_{\max} - W_{\min}}
$$

当 $W = W_{\min}$ 时，$q = 0$。因此零点为：

$$
z = \text{round}\left( \frac{-W_{\min}}{s} \right)
$$

#### 5.2.1.2 对称量化

**定义**：量化范围关于零对称。

**数学表达**：

$$
s = \frac{\max(|W|)}{2^{b-1} - 1}
$$

$$
q = \text{round}\left( \frac{W}{s} \right)
$$

$$
\hat{W} = s \cdot q
$$

**优点**：零点为 0，简化计算。

**缺点**：对于非对称分布的数据，量化误差较大。

### 5.2.2 量化误差分析

#### 5.2.2.1 量化误差的定义

$$
\epsilon = W - \hat{W}
$$

#### 5.2.2.2 均匀量化误差的方差

**推导**：假设量化误差在 $[-s/2, s/2]$ 上均匀分布，则方差为：

$$
\text{Var}(\epsilon) = \frac{s^2}{12}
$$

代入 $s = \frac{W_{\max} - W_{\min}}{2^b - 1}$：

$$
\text{Var}(\epsilon) = \frac{(W_{\max} - W_{\min})^2}{12(2^b - 1)^2}
$$

**结论**：量化误差的方差与 $2^{2b}$ 成反比。每增加 1bit，误差降低为原来的 1/4。

#### 5.2.2.3 信噪比（SNR）

$$
\text{SNR} = 10 \log_{10} \frac{\text{Var}(W)}{\text{Var}(\epsilon)}
$$

代入 $\text{Var}(\epsilon) = \frac{s^2}{12}$：

$$
\text{SNR} \approx 6.02b + \text{const}
$$

**结论**：每增加 1bit，SNR 提升约 6dB。

### 5.2.3 非均匀量化

#### 5.2.3.1 对数量化

**定义**：按对数尺度量化。

**数学表达**：

$$
q = \text{round}\left( \frac{\log(|W| + \epsilon) - \log(\epsilon)}{s} \right)
$$

**优点**：对小值更敏感，适合权重的长尾分布。

#### 5.2.3.2 NF4 量化

NF4（NormalFloat 4-bit）是 QLoRA 中使用的量化格式，专门为正态分布的权重设计。

**数学表达**：

$$
q_i = \Phi^{-1}\left( \frac{i + 0.5}{2^b} \right)
$$

其中 $\Phi^{-1}$ 是标准正态分布的逆累积分布函数。

**优点**：在权重密集的区域（0 附近）分配更多量化级别，量化误差更小。

## 5.3 量化的分类

### 5.3.1 按量化粒度分类

#### 5.3.1.1 逐层量化（Per-layer）

整个层使用一个缩放因子。

#### 5.3.1.2 逐通道量化（Per-channel）

每个通道使用一个缩放因子。

#### 5.3.1.3 逐组量化（Per-group）

每组权重使用一个缩放因子，组大小通常为 32、64、128。

### 5.3.2 按量化时机分类

#### 5.3.2.1 训练后量化（Post-Training Quantization, PTQ）

先训练完整模型，再量化。

#### 5.3.2.2 量化感知训练（Quantization-Aware Training, QAT）

在训练过程中模拟量化，让模型适应量化误差。

### 5.3.3 按量化数值分类

#### 5.3.3.1 对称量化 vs 非对称量化

| 类型       | 零点       | 优点           | 缺点               |
| ---------- | ---------- | -------------- | ------------------ |
| 对称量化   | $z = 0$    | 计算简单       | 对非对称分布误差大 |
| 非对称量化 | $z \neq 0$ | 适应非对称分布 | 计算稍复杂         |

#### 5.3.3.2 静态量化 vs 动态量化

| 类型     | 缩放因子   | 优点     | 缺点         |
| -------- | ---------- | -------- | ------------ |
| 静态量化 | 训练时确定 | 推理快   | 需要校准数据 |
| 动态量化 | 推理时计算 | 无需校准 | 推理稍慢     |

## 5.4 训练后量化（PTQ）

### 5.4.1 定义

训练后量化是在模型训练完成后直接进行量化，不需要重新训练。

### 5.4.2 核心方法

#### 5.4.2.1 简单 PTQ

直接对权重进行量化，不需要校准数据。

**流程**：

```
步骤1：训练完整模型
步骤2：对权重进行量化
步骤3：部署量化模型
```

**优点**：简单、快速。

**缺点**：量化误差较大，性能可能下降较多。

#### 5.4.2.2 校准 PTQ

使用少量校准数据，确定最优的量化参数。

**流程**：

```
步骤1：训练完整模型
步骤2：用校准数据运行模型，收集激活值分布
步骤3：根据激活值分布确定量化参数
步骤4：量化权重和激活值
步骤5：部署量化模型
```

**校准方法**：

- **Min-Max**：使用激活值的最小值和最大值。
- **Moving Average Min-Max**：使用滑动平均。
- **KL 散度**：最小化量化前后分布的 KL 散度。
- **MSE**：最小化量化前后的均方误差。

### 5.4.3 数学推导

#### 5.4.3.1 KL 散度校准

**目标**：找到最优的裁剪阈值 $T$，使得量化前后的分布 KL 散度最小。

$$
\min_T \ \text{KL}\left( p(W) \parallel p(\hat{W}) \right)
$$

其中 $p(W)$ 是原始分布，$p(\hat{W})$ 是量化分布。

**KL 散度的计算**：

将原始分布和量化分布都离散化为直方图，然后计算：

$$
\text{KL} = \sum_{i} p_i \log \frac{p_i}{q_i}
$$

其中 $p_i$ 是原始直方图的第 $i$ 个 bin，$q_i$ 是量化直方图的第 $i$ 个 bin。

#### 5.4.3.2 MSE 校准

**目标**：找到最优的量化参数，使得量化前后的 MSE 最小。

$$
\min_{s, z} \ \mathbb{E}\left[ (W - \hat{W})^2 \right]
$$

对于对称量化，最优的缩放因子为：

$$
s^* = \arg\min_s \ \mathbb{E}\left[ (W - s \cdot \text{round}(W/s))^2 \right]
$$

### 5.4.4 优缺点

**优点**：

- 不需要重新训练，计算成本低。
- 适用于大规模模型。
- 实现简单。

**缺点**：

- 性能损失可能较大（尤其对于低比特量化）。
- 需要校准数据。
- 对激活值的量化误差较敏感。

## 5.5 量化感知训练（QAT）

### 5.5.1 定义

量化感知训练是在训练过程中模拟量化操作，让模型适应量化误差。

### 5.5.2 核心思想

在前向传播中插入伪量化操作：

$$
\hat{W} = \text{fake\_quant}(W) = s \cdot \text{round}\left( \frac{\text{clip}(W, -T, T)}{s} \right)
$$

反向传播时，使用直通估计器（Straight-Through Estimator, STE）传递梯度。

### 5.5.3 直通估计器（STE）

#### 5.5.3.1 问题

量化函数 $q = \text{round}(W/s)$ 的梯度几乎处处为零：

$$
\frac{\partial q}{\partial W} = 0
$$

这导致反向传播时梯度无法通过量化层。

#### 5.5.3.2 解决方案

STE 将量化函数的梯度近似为 1：

$$
\frac{\partial \hat{W}}{\partial W} \approx 1
$$

**数学表达**：

在前向传播中：

$$
\hat{W} = s \cdot \text{round}\left( \frac{\text{clip}(W, -T, T)}{s} \right)
$$

在反向传播中：

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \hat{W}} \cdot \frac{\partial \hat{W}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial \hat{W}}
$$

#### 5.5.3.3 STE 的偏差分析

STE 引入的梯度偏差为：

$$
\text{Bias} = \frac{\partial \mathcal{L}}{\partial \hat{W}} - \frac{\partial \mathcal{L}}{\partial W}
$$

虽然 STE 是有偏的，但在实践中效果良好。

### 5.5.4 数学推导

#### 5.5.4.1 QAT 的损失函数

$$
\mathcal{L} = \mathcal{L}_{\text{task}}(f_{\hat{W}}(x), y)
$$

其中 $\hat{W}$ 是伪量化后的权重。

#### 5.5.4.2 梯度传播

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \hat{W}} \cdot \frac{\partial \hat{W}}{\partial W}
$$

使用 STE：

$$
\frac{\partial \hat{W}}{\partial W} \approx \begin{cases} 1 & \text{if } -T \leq W \leq T \\ 0 & \text{otherwise} \end{cases}
$$

### 5.5.5 优缺点

**优点**：

- 性能损失小（通常 <1%）。
- 可以实现低比特量化（如 INT4）。
- 模型可以适应量化误差。

**缺点**：

- 需要重新训练，计算成本高。
- 需要完整的训练数据和流程。
- 实现比 PTQ 复杂。

### 5.5.6 代码示例

```python
class FakeQuantize(nn.Module):
    """伪量化模块"""
    def __init__(self, bits=8, symmetric=True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
    
    def forward(self, x):
        if self.symmetric:
            # 对称量化
            scale = x.abs().max() / (2 ** (self.bits - 1) - 1)
            x_quant = torch.round(x / scale) * scale
        else:
            # 非对称量化
            x_min, x_max = x.min(), x.max()
            scale = (x_max - x_min) / (2 ** self.bits - 1)
            zero_point = torch.round(-x_min / scale)
            x_quant = (torch.round(x / scale) + zero_point) * scale
        
        # STE: 前向传播用量化值，反向传播梯度直接传递
        return x + (x_quant - x).detach()

class QATModel(nn.Module):
    """量化感知训练的模型"""
    def __init__(self, base_model, bits=8):
        super().__init__()
        self.base_model = base_model
        self.quant = FakeQuantize(bits)
    
    def forward(self, x):
        # 量化权重
        for name, param in self.base_model.named_parameters():
            if 'weight' in name:
                param.data = self.quant(param.data)
        
        return self.base_model(x)
```

## 5.6 大模型量化

### 5.6.1 GPTQ

#### 5.6.1.1 核心思想

GPTQ（Generative Pre-trained Transformer Quantization）是一种针对 LLM 的后训练量化方法。其核心思想是：逐层量化权重，并用校准数据最小化量化误差。

#### 5.6.1.2 数学推导

**目标函数**：

$$
\min_{\hat{W}} \ \| W X - \hat{W} X \|_F^2
$$

其中 $W$ 是原始权重，$\hat{W}$ 是量化后的权重，$X$ 是校准数据。

**逐列量化**：

将 $W$ 的每一列分别量化。对于第 $i$ 列 $w_i$：

$$
\min_{\hat{w}_i} \ \| w_i X - \hat{w}_i X \|_2^2
$$

**Hessian 矩阵**：

设 $H = 2XX^\top$，则最优量化误差为：

$$
\hat{w}_i = \text{quantize}\left( w_i - \frac{1}{H_{ii}} (w_i - \hat{w}_i) H_{:,i} \right)
$$

**推导**：利用 OBS（Optimal Brain Surgeon）的思想，量化第 $i$ 列时的误差补偿为：

$$
\delta_i = -\frac{w_i - \hat{w}_i}{H_{ii}} H_{:,i}
$$

这表示量化第 $i$ 列后，通过调整其他列来补偿误差。

#### 5.6.1.3 代码示例

```python
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",
    group_size=128,
    desc_act=False
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 5.6.2 AWQ

#### 5.6.2.1 核心思想

AWQ（Activation-aware Weight Quantization）的核心观察是：**不是所有权重都同等重要，应该保护那些对激活值影响大的权重。**

#### 5.6.2.2 数学推导

**缩放变换**：

$$
W' = W \cdot \text{diag}(s), \quad X' = X \cdot \text{diag}(s)^{-1}
$$

使得 $W'X' = WX$。

**缩放因子的确定**：

$$
s_j = \left( \frac{\max(|X_j|)}{\max(|W_j|)} \right)^\alpha
$$

其中 $\alpha$ 是超参数，通常取 0.5。

**为什么有效**：

- 对于激活值大的通道，增大缩放因子 $s_j$，使 $W_j$ 变小。
- 小权重的量化误差更小。
- 同时，$X_j$ 被缩小，保持乘积不变。

### 5.6.3 GGUF

#### 5.6.3.1 定义

GGUF（GPT-Generated Unified Format）是一种专为 LLM 设计的量化格式，支持多种量化级别（Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_0 等）。

#### 5.6.3.2 量化级别

| 级别 | 比特数 | 说明                   |
| ---- | ------ | ---------------------- |
| Q2_K | 2-3    | 极低精度，性能损失较大 |
| Q3_K | 3-4    | 低精度                 |
| Q4_K | 4-5    | 常用，平衡性能和质量   |
| Q5_K | 5-6    | 较高质量               |
| Q6_K | 6-7    | 高质量                 |
| Q8_0 | 8      | 接近原始精度           |

### 5.6.4 SmoothQuant

#### 5.6.4.1 核心思想

SmoothQuant 的核心思想是：**激活值比权重更难量化**。通过将量化难度从激活值转移到权重，实现更好的量化效果。

#### 5.6.4.2 数学推导

**缩放变换**：

$$
Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)
$$

其中 $s$ 是缩放因子。

**缩放因子的确定**：

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}
$$

其中 $\alpha$ 是超参数，通常取 0.5。

**效果**：激活值的范围被缩小，权重的范围被放大，使得两者都更容易量化。

## 5.7 量化与蒸馏、剪枝的结合

### 5.7.1 蒸馏 + 量化

**流程**：

```
步骤1：训练大模型（教师）
步骤2：蒸馏得到学生模型
步骤3：对学生模型进行量化
步骤4：部署量化后的学生模型
```

**联合优化**：

在蒸馏损失中加入量化误差项：

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{KD}} + (1 - \alpha) \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{quant}}
$$

其中 $\mathcal{L}_{\text{quant}}$ 是量化误差。

### 5.7.2 剪枝 + 量化

**流程**：

```
步骤1：训练完整模型
步骤2：剪枝得到稀疏模型
步骤3：量化剪枝后的模型
步骤4：微调恢复性能
```

**注意**：剪枝后的稀疏模型可能更难量化，因为非零权重的分布可能不均匀。

### 5.7.3 三者联合优化

**完整流程**：

```
原始模型 → 知识蒸馏（架构压缩）→ 剪枝（参数压缩）→ 量化（精度压缩）→ 部署
```

**示例**：

| 阶段     | 模型大小 | 说明     |
| -------- | -------- | -------- |
| 原始模型 | 70B      | FP16     |
| 蒸馏后   | 7B       | FP16     |
| 剪枝后   | 5B       | 30% 稀疏 |
| 量化后   | 2.5B     | INT8     |
| 总压缩率 | 28×      | —        |

### 5.7.4 联合优化的挑战

| 挑战     | 说明                  | 缓解方法           |
| -------- | --------------------- | ------------------ |
| 误差累积 | 每步压缩都引入误差    | 每步后微调         |
| 顺序依赖 | 不同顺序效果不同      | 实验确定最优顺序   |
| 硬件限制 | 稀疏+量化需要特殊硬件 | 使用支持的硬件     |
| 性能损失 | 多次压缩导致性能下降  | 逐步压缩，监控性能 |
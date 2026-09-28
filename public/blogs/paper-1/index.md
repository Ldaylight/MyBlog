title: "模型压缩和知识蒸馏的语音模型"
aliases:
  - "Efficient Speech Translation through Model Compression and Knowledge Distillation"
tags:
  - 模型压缩
  - 知识蒸馏
  - 语音翻译
  - QLoRA
  - 层剪枝
  - Qwen2-Audio
  - IWSLT-2025
created: 2026-09-17
source: "arXiv:2505.20237v2；IWSLT 2025 Model Compression track"
author: "Yasmin Moslem"
year: 2025
theme: "以 QLoRA 4-bit 量化、迭代层剪枝和序列级知识蒸馏压缩 Qwen2-Audio-7B-Instruct，用于英德与英中语音翻译。"
study_area: "语音翻译；多模态大模型压缩；高效模型部署"
data_source: "ACL 60/60（域内）；CoVoST2（域外补充）；Qwen2-Audio-7B-Instruct"
methodology: "全参数微调、序列级知识蒸馏、迭代 decoder 层剪枝、QLoRA 4-bit 量化与多阶段恢复微调"
core_variable: "压缩率（层数、参数量、存储）与翻译质量（BLEU、chrF/chrF++、COMET）及推理速度之间的权衡"
key_finding: "QLoRA 加知识蒸馏可在超过 40% 压缩下提高翻译质量；解码器迭代剪枝、QLoRA、知识蒸馏和多阶段微调组合可达到约 50% 压缩，并保留教师模型 97%–100% 的翻译质量。"
relevance: "提供了剪枝、量化和知识蒸馏组合用于语音模型的实验证据，并明确指出存储优化与推理延迟优化并不总是一致。"
type: literature-note
zotero_key: "AQIFVCXT"
pdf_key: "GKJVZMA5"
doi: ""
collection: "智能语音"
note_path: "note/智能语音/模型压缩和知识蒸馏的语音模型.md"
fulltext_path: "03fulltext/智能语音/AQIFVCXT.md"

# 基本信息

| 字段          | 内容                                                                   |
| ----------- | -------------------------------------------------------------------- |
| Zotero 条目   | [打开条目](zotero://select/library/items/AQIFVCXT)                       |
| 原始 PDF      | [打开 PDF](zotero://open-pdf/library/items/GKJVZMA5)                   |
| 全文 Markdown | [[03fulltext/智能语音/AQIFVCXT|正式 MinerU 全文]]；`page_mapping: reliable`，覆盖 10/10 页，无图片引用或缺失图片 |
| 作者 / 年份     | Yasmin Moslem / 2025                                                 |
| 来源          | IWSLT 2025 “Model Compression” track 系统报告；arXiv:2505.20237v2         |
| 分类          | 智能语音                                                                 |
| 身份键         | `zotero_key = AQIFVCXT`；`pdf_key = GKJVZMA5`                         |

Zotero 父条目只保存了中文题名，作者、年份和 arXiv 信息由原始 PDF 首页核验补入。Zotero 中没有 Creator、Tag、批注或独立笔记。

# 一句话摘要

论文在 IWSLT 2025 模型压缩赛道中，以 Qwen2-Audio-7B-Instruct 为基座，比较“QLoRA 4-bit + 序列级知识蒸馏”和“解码器迭代层剪枝 + 多阶段微调 + QLoRA + 知识蒸馏”两条路线；前者以较温和的压缩显著提升翻译质量，后者将参数和存储压缩约一半，同时保留教师模型约 97%–100% 的英中、英德翻译质量。

# 研究对象

论文研究的不是一个通用“语音识别模型”，而是语音翻译用的音频语言模型：

- 基座模型：Qwen2-Audio-7B-Instruct。
- 模型结构：音频编码器 `audio_tower` 和文本生成解码器 `language_model` 各 32 层。
- 原始规模：Table 1 和 Table 2 记为 8.40B 参数、16.79 GB 模型文件。
- 任务：英语语音翻译为德语和中文，即 EN-DE 与 EN-ZH。
- 约束：参赛系统必须从 Qwen2-Audio 派生；ACL 60/60 是约束赛道要求的域内数据。
- 核心问题：能否通过剪枝、低比特量化和知识蒸馏，在显著降低参数量与存储占用后，仍保留接近教师模型的语音翻译质量。

# 研究方法（方法概述）

作者先用 ACL 60/60 对基座模型做全参数微调，得到一个域内教师模型，再比较两条压缩路线。

1. **质量优先路线**：用教师模型生成的硬标签数据做序列级知识蒸馏，对教师模型做 4-bit QLoRA 微调，该路线不改变层数，但显著减少参数与存储。
2. **高压缩路线**：对教师模型的解码器做迭代层剪枝，每次删除对 chrF/chrF++ 伤害最小的 1 层；主实验删除 8 层，使解码器从 32 层降为 24 层，同时保留全部 32 层音频编码器。随后用ACL + 硬标签数据做全参微调，之后依次做QLoRA、知识蒸馏和 CoVoST2 域外数据恢复。
3. **消融比较**：作者比较 Base 与 Instruct、编码器加解码器剪枝与仅解码器剪枝、迭代剪枝与固定中间层剪枝、chrF/chrF++ 与 COMET 作为重要性指标、删除 8/10/12/16 层、剪枝后恢复与逐层立即恢复，以及 10k–100k CoVoST2 数据规模。

这里加入两个流程图方便理解：

路线一：

![image-20260927233506911](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260927233506911.png)

路线二：

![image-20260927233549951](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260927233549951.png)

## 方法分析

**教师模型和知识蒸馏**

教师模型先在 ACL 60/60 上全参数微调 3 个 epoch，batch size 为 4，学习率为 `1e-5`，weight decay 为 `0.001`，无 warm-up，模型以 bfloat16 加载。之后，作者用教师翻译 ACL 60/60 训练集，把伪标签与真实数据合并并去重，形成序列级知识蒸馏数据。

这里的知识蒸馏是“教师生成序列 + 学生监督微调”的 sequence-level KD。其优势是工程简单、可直接复用翻译数据；代价是学生学到的信息上限受教师输出质量限制，而且伪标签过滤与去重会显著影响数据规模和覆盖度。

**QLoRA**

作者通过 BitsAndBytes 将模型量化为 4-bit NF4，并启用 `double_quant`；LoRA rank 为 64，alpha 为 128，dropout 为 0，作用于所有线性模块，同时使用 rsLoRA。该配置仅训练 2.41% 的参数，训练 4 个 epoch，batch size 为 4，学习率为 `1e-5`。

这种设置优化的是“可训练参数和模型文件大小”，而不是必然优化延迟。论文后文明确指出，4-bit QLoRA 在降低存储的同时会付出推理速度代价。

**迭代层剪枝**

作者把解码器的每一层逐个移除，测量剩余模型的翻译质量，再删除损失最小的一层；随后在新模型上重复评估，直到删除目标层数。最终删除的层并不集中在中间位置：

- 德语：`[1, 3, 9, 13, 19, 20, 27, 29]`
- 中文：`[3, 4, 7, 15, 20, 25, 26, 29]`

这说明不同语言的“关键层”不同，固定删除中间 8 层并不是可靠替代方案。作者还发现首层或末层等位置对翻译质量尤其敏感，因此逐层搜索比静态层选择更稳健。

**多阶段恢复与最终训练**

剪枝后的模型先在 ACL 60/60 加教师知识蒸馏数据上微调 4 个 epoch；随后再做 1 个 epoch QLoRA，训练数据由 ACL 60/60、放大 10 倍的知识蒸馏数据和 100k CoVoST2 语句组成，batch size 为 8，学习率为 `1e-5`。多阶段训练的作用是先修复剪枝造成的表示损伤，再用域外数据补充泛化能力。

**关键消融结论**

- Instruct 模型显著优于 Base 模型，因此后续均使用 Instruct。
- 只剪解码器优于同时剪编码器和解码器。
- 迭代剪枝明显优于固定中间层剪枝。
- 用 chrF/chrF++ 评估层重要性，最终效果优于用 COMET 评估。
- 删除 8、10、12 层后恢复效果接近；删除 16 层后质量明显下降。
- 每删一层就立即微调反而没有优于“全部剪完后再微调”，且计算成本更高。
- 域外数据从 10k 增至 50k、80k 总体改善明显，继续增加到 100k 后 BLEU 和 chrF++ 出现边际收益递减，但 COMET 仍有提升。

# 数据来源

|          数据或模型          |   作用    |                         规模与划分                         |
| :---------------------: | :-----: | :---------------------------------------------------: |
|        ACL 60/60        | 域内训练与测试 |   合并 dev/eval 后共 884 条；随机抽取 100 条测试、784 条训练，随机种子为 0   |
|          教师伪标签          | 知识蒸馏数据  |               去重后德语 1,568 段、中文 1,069 段                |
|         CoVoST2         | 域外恢复数据  | 英语到 15 种语言，包括德语和中文；消融测试 10k、50k、80k、100k，最终主实验使用 100k |
| Qwen2-Audio-7B-Instruct | 基座与教师模型 |                8.40B 参数、16.79 GB 模型文件                 |

评价指标为 BLEU、chrF++/chrF 和 COMET；德语使用 chrF++，中文使用原始 chrF；COMET 使用 `wmt20-comet-da`。推理时关闭采样，采用 greedy generation，batch size 为 1，最大生成长度为 1,024 tokens。

数据层面的主要限制是域内训练集只有 784 条、测试集只有 100 条，且 ACL 60/60 是学术演讲领域的受限数据。CoVoST2 虽然扩大了语言和语音覆盖，但其域外数据与目标测试域并不相同。

# 研究结论

## 主要发现 1：QLoRA 加知识蒸馏是“质量优先”的强压缩方案

**发现**：在不解码器剪枝的情况下，QLoRA 4-bit 与序列级知识蒸馏组合将模型压缩超过 40%，并在英德、英中两组成绩上超过未压缩的原始 Qwen2-Audio 基线和普通全参数微调模型。

**原文引用**：

> “the combination of QLoRA with 4-bit quantization and knowledge distillation has achieved the highest translation performance into both German and Chinese across all evaluation metrics while reducing the model size in terms of both the number of parameters and storage requirements by more than 40% compared to the baseline model.”

**证据位置**：PDF 第 3 页；Table 1 位于 PDF 第 2 页。

## 主要发现 2：剪枝、QLoRA 和知识蒸馏组合可达到约 50% 压缩

**发现**：解码器删除 8 层后，模型参数从 8.40B 降到 4.12B，存储从 16.79 GB 降到 8.65 GB；结合 4-bit QLoRA、知识蒸馏和 CoVoST2 微调后，最终质量约为教师模型的 97%（中文）和 100%（德语）。

**原文引用**：

> “The whole process of pruning followed by QLoRA fine-tuning with 4-bit quantization has resulted in approx. 50% reduction in the model size, while retaining 97% and 100% of the translation quality for Chinese and German, respectively, compared to the teacher model.”

**证据位置**：PDF 第 3 页，Table 2；完整配置分析见第 4 页。

## 主要发现 3：逐层重要性搜索优于固定中间层剪枝

**发现**：作者先测每层删除后的翻译表现，再逐层删除伤害最小的层。这个性能引导过程避免了“中间层一定不重要”的静态假设，最终删除层分布在模型前部、中部和后部。

**原文引用**：

> “After identifying and removing the least critical layer, we repeat the layer importance evaluation on the remaining layers until reaching our n pruning target.”

> “In iterative layer pruning based on performance evaluation ... the pruned layers are [1, 3, 9, 13, 19, 20, 27, 29] for German, and [3, 4, 7, 15, 20, 25, 26, 29] for Chinese. This shows a diverse layer selection that is not concentrated solely in the middle.”

**证据位置**：PDF 第 4、6 页。

## 主要发现 4：只剪解码器优于同时剪编码器和解码器

**发现**：英语到德语实验中，将解码器从 32 层剪到 24 层，优于把编码器和解码器都剪到 24 层；后者的 BLEU、chrF++ 和 COMET 均更低。

**原文引用**：

> “pruning only the decoder from 32 layers to 24 layers outperformed pruning both the encoder and decoder into 24 layers.”

**证据位置**：PDF 第 5 页，Table 3。

## 主要发现 5：迭代剪枝和 chrF/chrF++ 评估更有效

**发现**：删除同样 8 层时，逐层剪枝显著优于固定删除第 12–19 层；在层重要性指标上，chrF/chrF++ 训练出的最终模型优于 COMET。

**原文引用**：

> “Iterative layer pruning, i.e. removing layers one by one, and then evaluating the resulting model, outperforms middle-layer pruning. In particular, when chrF/chrF++ is used for layer importance evaluation, the resulting pruned model achieves better speech translation quality after fine-tuning on the ACL 60/60 dataset.”

**证据位置**：PDF 第 6 页，Table 5。

## 主要发现 6：剪枝存在明显的深度拐点

**发现**：删除 8、10、12 层后，经过 ACL 60/60 恢复微调的效果仍接近；继续删除至 16 层时，质量显著恶化。德语对剪枝比中文更敏感。

**原文引用**：

> “We observe that the quality after pruning up to 12 layers and fine-tuning the pruned model on the ACL 60/60 dataset is close to pruning 8 layers. However, when pruning 16 layers, the quality starts to degrade considerably.”

**证据位置**：PDF 第 7 页，Table 6。

## 主要发现 7：剪枝优化速度，4-bit 量化优化存储但可能拖慢推理

**发现**：删除 8 层和 16 层分别带来约 20% 和 40% 的推理加速；QLoRA 的 4-bit 量化减少存储，但产生推理速度代价。若部署目标是低延迟而非最小存储，作者建议使用标准 LoRA 而不是 QLoRA。

**原文引用**：

> “Pruning reduces storage footprint while accelerating inference speed by approximately 20% and 40% when 8 and 16 layers are pruned, respectively ... In contrast, 4-bit quantization as implemented in QLoRA reduces storage at the cost of inference speed.”

**证据位置**：PDF 第 7–8 页。

## 主要发现 8：质量最高的压缩路线与压缩率最高的路线不同

**发现**：QLoRA 加知识蒸馏的翻译质量优于单独依赖层剪枝，但压缩率低于“剪枝 + 量化 + 蒸馏 + 多阶段微调”的组合。因此，压缩方案应根据目标是质量上限还是更高压缩率来选择。

**原文引用**：

> “QLoRA fine-tuning with knowledge distillation achieved superior translation quality compared to layer pruning alone, though with reduced model compression. To achieve higher compression ratios while preserving translation quality, we employed a combined approach using iterative layer pruning, quantization, knowledge distillation, and multi-stage fine-tuning.”

**证据位置**：PDF 第 8 页。



# 关联精读笔记

- [[note/知识蒸馏/Distilling the Knowledge in a Neural Network|Distilling the Knowledge in a Neural Network]]：提供 soft targets、temperature softmax 和知识蒸馏的一般框架；本文使用的 teacher-generated sequence KD 与该方法在概念上相关，但监督形式不同。

- [[note/大模型微调/LoRA Low-Rank Adaptation of Large Language Models|LoRA: Low-Rank Adaptation of Large Language Models]]：提供冻结预训练权重、只训练低秩分解矩阵的参数高效适配机制；本文的 QLoRA 配置以此为底座。

- [[note/大模型微调/QLoRA Efficient Finetuning of Quantized LLMs|QLoRA: Efficient Finetuning of Quantized LLMs]]：提供 NF4、Double Quantization、Paged Optimizers 与 4-bit 量化基座上的 LoRA 微调方法；本文的 4-bit QLoRA 配置直接对应该方法。

# 我的判断

**这篇论文最有价值的地方**

1. 它没有把“压缩”简化成单一技术，而是把剪枝、量化、知识蒸馏和多阶段微调拆开做消融，清楚展示了不同组合在质量、参数、存储和速度上的取舍。
2. 层剪枝实验给出了具体删除层数和层号，并指出不同语言的层重要性不同；这比只报告平均压缩率更有复现和迁移参考价值。
3. 论文主动区分了“省存储”和“低延迟”：剪枝有利于推理速度，QLoRA 更有利于存储。这一点对实际部署比单纯的参数压缩数字更重要。

**需要谨慎看待的地方**

1. 研究对象只覆盖 Qwen2-Audio-7B-Instruct、英德和英中两个方向，不能直接外推到自动语音识别、语音理解或更多语言。
2. 域内训练集仅 784 条、测试集仅 100 条，且分布是科学演讲；跨说话人、口音、噪声和真实产品的泛化证据不足。
3. 论文对“推理速度提升”主要给文字结论，没有看到完整的延迟、吞吐、显存峰值和不同硬件对照表。报告中的 20% 和 40% 更适合视为该实验设置下的方向性结果，而不是通用部署保证。
4. 4-bit 量化的速度代价说明，压缩率、存储和延迟之间不存在自动同向关系。若目标是实时语音翻译，应单独测量解码吞吐和量化 kernel 开销。
5. 剪枝后的恢复依赖教师伪标签和 CoVoST2；在无法获得高质量教师输出或域外数据的场景中，97%–100% 的质量保持未必可复制。
6. 存储统计只计算模型 `*.safetensors` 文件，排除了 tokenizer 和配置等约 13 MB 的固定开销。对于极小模型或端侧多组件系统，这个口径需要重新核算。

**对我的可迁移启示**

- 做语音模型压缩时，优先选择 **decoder-only 迭代剪枝**，不要默认编码器和解码器可以同比例压缩。
- 层重要性应使用目标任务指标搜索；在这篇论文中，chrF/chrF++ 比 COMET 更适合做剪枝选择。
- 知识蒸馏适合和剪枝、量化组合作为恢复阶段，但不应把它视为提升压缩率本身的技术。
- 如果目标是低成本存储，可考虑 QLoRA 4-bit；如果目标是低延迟推理，应优先剪枝并谨慎评估量化开销，甚至改用标准 LoRA。
- 评估压缩模型时，至少同时报告参数量、模型文件大小、BLEU/chrF/COMET、延迟、吞吐和显存，否则容易把“文件更小”误判成“部署更高效”。

[^1]: 







# 组会问题——2026.09.22

## 问题1：为什么数据集这么小要进行全参数微调？

**全参数微调听起来需要大量数据，但实际上，对于一个已经预训练并指令微调过的大模型，用少量高质量领域数据做全参数微调是完全可行的**

小数据全参数微调大模型之所以可行，是因为模型已经预训练并指令微调过，具备翻译能力；微调只是让模型适应特定领域，而不是从零学习。

但是小数据集进行全参微调还会出现新的问题：**过拟合问题**

小数据全参数微调确实容易过拟合。论文也意识到这一点，所以：低学习率、少 epoch、高质量数据、正则化，共同防止了过拟合；后续用知识蒸馏、剪枝、QLoRA 等多阶段训练；用 CoVoST2 等外部数据恢复泛化。



## 问题2：Qwen2-Base模型和Qwen2-Instruct模型的区别？为什么使用Qwen2-Instruct模型？

**这两个模型都是 Qwen 官方提前发布好的，不是论文作者自己训练的。**

- **Qwen2-Audio-7B**：Qwen 官方发布的基座模型（base），只经过大规模预训练。
- **Qwen2-Audio-7B-Instruct**：Qwen 官方在基座模型上做了指令微调/对齐后发布的版本（instruct）。

论文作者只是**直接加载并使用**了这两个官方模型，先比较了 base 和 instruct；发现 instruct 明显更好；于是后续所有实验都基于 **Qwen2-Audio-7B-Instruct**

- **Base 模型**：只经过预训练，本质是“给定前文，预测下一个词”。你给它一段音频和文字，它可能继续往下写，但不一定按“翻译成德语”这个要求来做。输出格式和内容都不受控。
- **Instruct 模型**：在 base 基础上做了指令微调，甚至可能经过 RLHF/DPO 等对齐。它见过大量“指令-回答”数据，所以能理解“把这段语音翻译成中文”这类任务，并按要求输出。



## 问题3：解释 BLEU、chrF++ 和 COMET 这些评估指标的作用？

论文中使用了三个指标：

- **BLEU**：词级 n-gram 重叠；
- **chrF / chrF++**：字符级和词级 n-gram 重叠；
- **COMET**：基于神经网络的语义相似度。

论文同时报告三者，是为了让翻译质量评估更全面、更有说服力。

|    指标    |                         用了什么方法                         |                     能衡量什么                     |                             特点                             |
| :--------: | :----------------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------------------: |
|  **BLEU**  |             统计词级 n-gram 精确率，并加简短惩罚             |   候选译文和参考译文在**词和短语层面**的匹配程度   | 快、经典、通用；但只看表面词重叠，不懂语义，对中文和形态丰富语言不友好 |
|  **chrF**  |       统计字符级 n-gram 的精确率和召回率，计算 F 分数        |      候选译文和参考译文在**字符层面**的相似度      |    对德语、中文等更鲁棒；但仍是表面匹配，可能忽略词级错误    |
| **chrF++** |             在 chrF 基础上，额外加入词级 n-gram              |       同时衡量**字符层面 + 词层面**的相似度        | 比 chrF 更严格，能惩罚冠词、词形等错误；论文中德语用 chrF++  |
| **COMET**  | 用神经网络模型，输入源句、参考译文、候选译文，回归出质量分数 | **语义层面**的翻译质量，看候选是否忠实于源句和参考 | 更接近人工判断，能捕捉同义替换和语义等价；但计算更贵，依赖训练数据 |



### 3.1 BLEU

**BLEU**，双语评估替补。具体做法是统计 **n-gram 精确率**，即候选译文中的词组有多少也出现在参考译文里。

**N-gram**：连续N个词的重叠，通常取 1 到 4-gram，分别计算**精确率**，然后取几何平均。

#### 3.1.1 计算公式

对每个 n，计算 n-gram 精确率：

$$
p_n = \frac{\text{候选译文中与参考匹配的 n-gram 数}}{\text{候选译文中的 n-gram 总数}}
$$
BLEU 公式：
$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$
通常：

- N = 4
- $w_n = 1/N$，即每个 n-gram 权重相同；
- BP 是简短惩罚。

#### 3.1.2 简短惩罚

如果模型只输出一个很短的句子，但恰好都匹配，精确率会很高。比如参考是 10 个词，候选只输出 1 个词且匹配，精确率 100%。

所以 BLEU 加入简短惩罚：
$$
BP = 
\begin{cases}
1, & \text{如果 } c > r \\
\exp(1 - r/c), & \text{如果 } c \le r
\end{cases}
$$
其中：

- c：候选译文长度；
- r：参考译文长度。

这样模型不能靠“少说少错”来刷分。

#### 3.1.3 例子

**参考译文**：  `the cat is on the mat`

**候选译文**：  `the cat is on mat`

参考长度 $r = 6$ 个词，候选长度 $c = 5$ 个词。

#### 3.1.3.1 计算 n-gram 精确率

BLEU 通常取 1-gram 到 4-gram，分别计算修改后的精确率。

**1-gram**

候选词：`the, cat, is, on, mat`（共 5 个）  
参考词：`the, cat, is, on, the, mat`（共 6 个）

匹配数：`the` 1 个，`cat` 1 个，`is` 1 个，`on` 1 个，`mat` 1 个，共 5 个。

精确率：
$p_1 = \frac{5}{5} = 1.0$

**2-gram**

候选 2-gram：`the cat, cat is, is on, on mat`（共 4 个）  
参考 2-gram：`the cat, cat is, is on, on the, the mat`（共 5 个）

匹配：`the cat, cat is, is on`，共 3 个。

精确率：
$$p_2 = \frac{3}{4} = 0.75$$

**3-gram**

候选 3-gram：`the cat is, cat is on, is on mat`（共 3 个）  
参考 3-gram：`the cat is, cat is on, is on the, on the mat`（共 4 个）

匹配：`the cat is, cat is on`，共 2 个。

精确率：
$$p_3 = \frac{2}{3} \approx 0.6667$$

**4-gram**

候选 4-gram：`the cat is on, cat is on mat`（共 2 个）  
参考 4-gram：`the cat is on, cat is on the, is on the mat`（共 3 个）

匹配：`the cat is on`，共 1 个。

精确率：
$$p_4 = \frac{1}{2} = 0.5$$

#### 计算简短惩罚

候选长度 $c = 5$，参考长度 $r = 6$，因为 $c \le r$，所以：
$BP = \exp\left(1 - \frac{r}{c}\right) = \exp\left(1 - \frac{6}{5}\right) = \exp(-0.2) \approx 0.8187$

#### 计算 BLEU

BLEU 使用几何平均，通常 $N=4$，权重均匀 $w_n = 1/4$
$\text{BLEU} = 0.8187 \times 0.7071 \approx 0.579$

按百分制，BLEU ≈ **57.9**。

#### 优点

- 计算快；
- 经典、通用，方便和其他论文对比；
- 在新闻、正式文本等领域与人工判断有一定相关性。

#### 缺点

- 只看表面词重叠，不理解语义；
- 同义词、改写、语序变化会被误判；
- 对中文等需要分词的语言不友好；
- 参考译文只有一两个版本时，合理但不同的翻译会被扣分；
- 句子级别与人工判断相关性有限。



### 3.2 chrF 和 chrF++

**chrF** = Character n-gram F-score，字符 n-gram F 分数。

**chrF++** 是 chrF 的扩展版，额外加入词级 n-gram。

chrF 不看词，而是看**字符级 n-gram** 的匹配情况。

它同时计算：

- **精确率**：候选译文中的字符 n-gram 有多少出现在参考译文中；
- **召回率**：参考译文中的字符 n-gram 有多少出现在候选译文中；
- **F 分数**：精确率和召回率的调和平均。

#### 计算公式

精确率：
$$
P = \frac{\text{匹配的字符 n-gram 数}}{\text{候选译文中的字符 n-gram 总数}}
$$
召回率：
$$
R = \frac{\text{匹配的字符 n-gram 数}}{\text{参考译文中的字符 n-gram 总数}}
$$
F 分数：
$$
F_\beta = (1+\beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}
$$
通常 $\beta=2$ 或 3，更重视召回率。

#### chrF++ 多了什么

chrF++ 在字符 n-gram 基础上，**额外加入词级 n-gram**，同时考虑字符和词两个层面。

可以理解为：

- chrF：字符层面像不像？
- chrF++：字符层面 + 词层面像不像？



#### 例子

**参考译文**：  
`der Katze`

**候选译文**：  
`die Katze`

去掉空格：

- 参考字符序列：`d, e, r, K, a, t, z, e`，共 8 个字符。
- 候选字符序列：`d, i, e, K, a, t, z, e`，共 8 个字符。

chrF 通常取字符 n-gram 的 $n = 1$ 到 $6$，对每个 $n$ 计算精确率、召回率和 F 分数，然后取平均。

**1-gram**

参考字符：`d, e, r, K, a, t, z, e`  
候选字符：`d, i, e, K, a, t, z, e`

匹配字符：`d` 1 个，`e` 2 个，`K` 1 个，`a` 1 个，`t` 1 个，`z` 1 个，共 7 个匹配。

精确率：
$$P_1 = \frac{7}{8} = 0.875$$

召回率：
$$R_1 = \frac{7}{8} = 0.875$$

F 分数（取 $\beta=2$）：
$F_1 = (1+2^2) \cdot \frac{P_1 R_1}{2^2 P_1 + R_1} = 5 \cdot \frac{0.875 \times 0.875}{4 \times 0.875 + 0.875} = 0.875$

**2-gram**

参考 2-gram：`de, er, rK, Ka, at, tz, ze`，共 7 个。  
候选 2-gram：`di, ie, eK, Ka, at, tz, ze`，共 7 个。

匹配：`Ka, at, tz, ze`，共 4 个。

精确率：
$$P_2 = \frac{4}{7} \approx 0.5714$$

召回率：
$$R_2 = \frac{4}{7} \approx 0.5714$$

F 分数：
$$F_2 \approx 0.5714$$

**3-gram**

参考 3-gram：`der, erK, rKa, Kat, atz, tze`，共 6 个。  
候选 3-gram：`die, ieK, eKa, Kat, atz, tze`，共 6 个。

匹配：`Kat, atz, tze`，共 3 个。

精确率：
$$P_3 = \frac{3}{6} = 0.5$$

召回率：
$$R_3 = \frac{3}{6} = 0.5$$

F 分数：
$$F_3 = 0.5$$

**4-gram**

参考 4-gram：`derK, erKa, rKat, Katz, atze`，共 5 个。  
候选 4-gram：`dieK, ieKa, eKat, Katz, atze`，共 5 个。

匹配：`Katz, atze`，共 2 个。

精确率：
$$P_4 = \frac{2}{5} = 0.4$$

召回率：
$$R_4 = \frac{2}{5} = 0.4$$

F 分数：
$$F_4 = 0.4$$

**5-gram**

参考 5-gram：`derKa, erKat, rKatz, Katze`，共 4 个。  
候选 5-gram：`dieKa, ieKat, eKatz, Katze`，共 4 个。

匹配：`Katze`，共 1 个。

精确率：
$$P_5 = \frac{1}{4} = 0.25$$

召回率：
$$R_5 = \frac{1}{4} = 0.25$$

F 分数：
$$F_5 = 0.25$$

**6-gram**

参考 6-gram：`derKat, erKatz, rKatze`，共 3 个。  
候选 6-gram：`dieKat, ieKatz, eKatze`，共 3 个。

匹配：0 个。

F 分数：
$$F_6 = 0$$

**计算 chrF**

chrF 对 $n=1$ 到 $6$ 的 F 分数取平均：
$\text{chrF} = \frac{1}{6} \sum_{n=1}^{6} F_n$

代入：
$\text{chrF} = \frac{0.875 + 0.5714 + 0.5 + 0.4 + 0.25 + 0}{6} \approx \frac{2.5964}{6} \approx 0.4327$

按百分制，chrF ≈ **43.3**。

chrF++ 在 chrF 的字符级基础上，额外加入词级 n-gram 的 F 分数。



字符级部分与 chrF 完全相同，字符级平均分数为：
$$\text{chrF}_{\text{char}} \approx 0.4327$$



**词级部分**

参考词：`der, Katze`  
候选词：`die, Katze`

**词级 1-gram**

匹配：`Katze`，共 1 个。

精确率：
$$P_{\text{word1}} = \frac{1}{2} = 0.5$$

召回率：
$$R_{\text{word1}} = \frac{1}{2} = 0.5$$

F 分数：
$$F_{\text{word1}} = 0.5$$

**词级 2-gram**

参考：`der Katze`  
候选：`die Katze`

匹配：0 个。

F 分数：
$$F_{\text{word2}} = 0$$

词级平均：
$$\text{chrF}_{\text{word}} = \frac{0.5 + 0}{2} = 0.25$$

**合并**

简单平均（具体实现可能有权重）：
$$\text{chrF++} = \frac{\text{chrF}_{\text{char}} + \text{chrF}_{\text{word}}}{2} = \frac{0.4327 + 0.25}{2} \approx 0.3414$$

按百分制，chrF++ ≈ **34.1**。

**对比**

- chrF ≈ 43.3：只看字符重叠，忽略了词边界和词序，分数更高。
- chrF++ ≈ 34.1：同时看字符和词，惩罚了冠词错误，分数更低。



#### 优点

- 对形态变化丰富的语言更友好，比如德语、土耳其语；
- 对中文更合理，因为中文“词”的边界不清晰；
- 同时考虑精确率和召回率，能惩罚漏译。

#### 缺点

- 仍然主要看表面重叠，语义理解有限；
- 字符级匹配可能高估质量；
- 不同语言、不同分词方式可能影响结果；
- 对长句的语序变化仍然敏感。

#### 论文中的使用

论文中：德语用 **chrF++**；中文用 **chrF**。

原因是 chrF 作者指出：**中文里“词”的概念通常不清晰**，所以中文更适合纯字符级 chrF。



### 3.3 COMET

**COMET** = Crosslingual Optimized Metric for Evaluation of Translation，是一个**基于神经网络的语义评估指标**。是用一个预训练的多语言模型，同时输入：**源句**；**参考译文**；**候选译文**；然后输出一个分数，表示候选译文在语义上有多接近参考译文。

#### 工作流程

1. 用多语言预训练模型（如 XLM-R）分别编码源句、参考译文、候选译文；
2. 把三者表示拼接或交互；
3. 通过一个回归层输出质量分数；
4. 训练时用人工评分数据（如 WMT 的 DA 分数）做监督。

可以写成：
$$
\text{score} = f_\theta(x, y_{\text{ref}}, y_{\text{cand}})
$$
其中：

- $x$：源句；
- $y_{\text{ref}}$：参考译文；
- $y_{\text{cand}}$：候选译文；
- $f_\theta$：训练好的神经网络。

训练目标是最小化预测分数与人工评分的差距，通常用均方误差：
$$
L = (\text{score} - \text{human\_score})^2
$$

直观理解：BLEU/chrF 看“字面像不像”；COMET 看**“意思对不对”**。

#### 举例

例如：

- 源句：猫在垫子上。
- 参考：猫在垫子上（德语，带冠词）。
- 候选：猫在垫子上（德语，缺少冠词 `der`）。

虽然候选缺少一个冠词，但语义完全正确。COMET 会给一个较高的分数，因为它理解语义等价，不会因为缺少一个词就大幅扣分。

论文中，COMET 分数有正有负，例如 -44.44、39.08、59.21。这是因为 COMET 输出的是连续值，可能经过缩放。分数越高越好。



#### 优点

- 与人工判断相关性高；
- 能捕捉语义等价和改写；
- 适合评估整体翻译质量；

#### 缺点

- 需要额外模型推理，计算更贵；
- 依赖训练数据，可能有偏差；
- 对某些语言对或领域可能不够准确；







## 问题4：这里的学生模型是哪个？好像没有设置两个模型？

这篇论文里的知识蒸馏，**不是传统意义上**“用大模型直接训练一个架构不同的小模型”，而是直**接把教师模型压缩（剪枝 + 量化 + LoRA）**，得到**同架构但更小**的学生模型，再用教师生成的翻译数据对学生模型做序列级知识蒸馏。

也就是说，**压缩靠剪枝和 QLoRA，蒸馏靠教师生成伪标签数据**，两者是配合关系.它的“学生模型”其实就是**教师模型经过压缩后的版本**，蒸馏是通过**数据层面**实现的，而不是通过换模型实现的。

教师模型 = **Qwen2-Audio-7B-Instruct + ACL 60/60 全参数微调**。

学生模型不是新架构，而是教师模型经过压缩操作得到的：

- **路线一**：教师模型 → 4-bit 量化 → 挂 LoRA → 微调。
  参数量名义上还是 7B，但存储和显存降低约 40%。

![image-20260927233506911](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260927233506911.png)

- **路线二**：教师模型 → 剪掉 8 层 decoder（32→24）→ 全参微调 → 4-bit 量化 → 挂 LoRA → 微调。
  参数和存储降低约 50%。

![image-20260927233549951](https://cdn.jsdelivr.net/gh/Ldaylight/typora-image-bed//Typoraimage-20260927233549951.png)





## 问题5：Q-lora到底压缩了什么？

4-bit量化的不是LoRA参数(B A矩阵)，而是对预训练模型的参数进行量化的。

**主干权重 **：被 4-bit 量化，然后冻结，不参与训练

**LoRA 适配器 **：随机初始化，训练时只更新它们

|    层面     | 是否压缩 |            怎么实现             |
| :---------: | :------: | :-----------------------------: |
|   参数量    |  不压缩  |   模型还是 7B 参数，结构不变    |
| 存储 / 显存 |   压缩   | 原始权重 4-bit 量化，约降到 1/4 |
| 可训练参数  | 大幅压缩 |       只训 LoRA，约 2.41%       |
|  训练成本   | 大幅压缩 |      冻结主干，只更新 LoRA      |
|  推理速度   | 可能变慢 |     4-bit 反量化有额外开销      |


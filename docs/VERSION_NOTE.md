# v1：Learnable Span Query 与单层 Span-Text Cross-Attention 版本

## 一、版本定位

本分支不是纯原始 MESM baseline，而是在 MESM 基础上加入了两个轻量改进：

1. 为 DETR decoder 的 span query 增加可学习语义内容；
2. 在 decoder 第一层加入一次 span-text cross-attention。

因此，本版本可以作为后续 Pathformer 改进的基础版本。

整体流程为：

    enhance_encoder
        -> t2v_encoder
        -> DETR Transformer Encoder
        -> DETR Decoder
            - learnable span query content
            - first-layer span-text cross-attention
        -> 时刻定位 / 显著性预测

## 二、核心改进一：Learnable Span Query Content

原始 query_embed 主要表示时间参考点，即 span 的 center 和 width。

本版本额外加入 query_content_embed，用于给每个 span query 提供可学习的语义内容。

这样 decoder query 不再只是时间位置参考，而是同时具有：

    时间参考点
        +
    可学习语义内容

该设计有助于 decoder 在进行视频 memory cross-attention 之前具备更强的 query 表达能力。

## 三、核心改进二：单层 Span-Text Cross-Attention

本版本在 DETR decoder 第一层加入 span-text cross-attention。

具体思想是：

    span query 先看文本 token
        -> 得到文本引导的 span query
        -> 再看视频 memory

也就是说，decoder 的执行顺序可以理解为：

    learnable span content
        -> text cross-attention
        -> enhanced-video cross-attention

本版本只在第一层 decoder 注入文本信息，避免多层重复注入导致过强文本干扰。

## 四、保留内容

本版本仍然保留 MESM 原始主体结构：

1. enhance_encoder 保留；
2. t2v_encoder 保留；
3. 正负样本对比学习逻辑保留；
4. FW masked word reconstruction 保留；
5. SS reconstruction 保留；
6. 负样本仍然使用 batch 内其他文本作为完整负文本。

## 五、与 v2 的区别

v1 不包含 Pathformer AMS。

本版本的重点是：

    span query 语义增强
        +
    decoder 第一层文本引导

v2 则在此基础上进一步加入：

    DETR Transformer Encoder 后 Pathformer AMS 多尺度建模

## 六、主要修改点

主要涉及文件：

- model/model.py
- model/transformer.py

## 七、实验作用

v1 用于验证：

    learnable span query content
    和
    单层 span-text cross-attention

是否能够提升 MESM 的文本引导 span 定位能力。

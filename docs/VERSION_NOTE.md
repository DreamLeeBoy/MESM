# v2：DETR Transformer Encoder 后接 Pathformer AMS 版本

## 一、版本定位

本分支是在 v1 的基础上继续引入 Pathformer AMS 模块。

v1 已经包含：

1. learnable span query content；
2. decoder 第一层 span-text cross-attention。

v2 在此基础上，将 Pathformer AMS 插入到：

    DETR Transformer Encoder 之后
    DETR Decoder 之前

整体流程为：

    enhance_encoder
        -> t2v_encoder
        -> DETR Transformer Encoder
        -> Pathformer AMS
        -> DETR Decoder
            - learnable span query content
            - first-layer span-text cross-attention
        -> 时刻定位 / 显著性预测

## 二、核心改进：Encoder 后 Pathformer AMS

v2 的主要改进是对 DETR Transformer Encoder 输出的视频 memory 进行多尺度建模。

具体地，Transformer Encoder 输出 memory 后，先拆分为：

    global token memory
    local video memory

Pathformer 只作用在 local video memory 上，global token 保持不变。

处理过程为：

    encoded video memory
        -> local memory
        -> Pathformer AMS
        -> enhanced local memory
        -> DETR Decoder

## 三、Pathformer AMS 的作用

Pathformer AMS 用于自适应多尺度建模。

它通过多个不同 patch size 的 expert 建模不同时间尺度，例如：

    短时间片段
    中等时间片段
    长时间片段

同时通过 noisy top-k routing 自适应选择专家路径。

因此，v2 希望增强：

1. 编码后视频 memory 的多尺度时序表示；
2. 局部时间块内部关系；
3. 不同时间块之间的全局关系；
4. 不同 patch size 下的视频结构表达。

## 四、Pathformer 的插入位置

本版本的 Pathformer 插入位置是：

    Transformer Encoder 后
    Transformer Decoder 前

也就是说，Pathformer 不替换 enhance_encoder，也不直接改变 FW 分支。

它主要作为 encoder memory 的后处理增强模块。

## 五、保留内容

v2 保留 v1 的以下设计：

1. learnable span query content；
2. decoder 第一层 span-text cross-attention；
3. 原始 enhance_encoder；
4. t2v_encoder；
5. 正负样本对比学习；
6. FW masked word reconstruction；
7. SS reconstruction；
8. 原始 batch 完整文本负样本逻辑。

## 六、与 v1 的区别

v1：

    enhance_encoder
        -> t2v_encoder
        -> DETR Transformer Encoder
        -> DETR Decoder

v2：

    enhance_encoder
        -> t2v_encoder
        -> DETR Transformer Encoder
        -> Pathformer AMS
        -> DETR Decoder

因此，v2 相比 v1 的核心区别是：

    在 DETR Encoder 输出的视频 memory 上加入 Pathformer AMS 多尺度建模。

## 七、与已放弃 v3 思路的区别

v2 不将 Pathformer 移入 enhance_encoder。

也就是说，v2 不改变：

    正样本 enhance_encoder
    负样本 enhance_encoder
    FW enhance_encoder

v2 的 Pathformer 只作用于 DETR Transformer Encoder 后的视频 memory。

因此，v2 结构更清晰，风险更低，适合作为当前保留版本。

## 八、主要修改文件

主要涉及文件：

- model/pathformer_encoder.py
- model/transformer.py
- model/model.py
- model/criterion.py
- runner.py
- utils/config.py
- config/charades/C+SF_C.json

## 九、实验作用

v2 用于验证：

    在 v1 的 span query 文本引导基础上，
    进一步加入 Encoder 后 Pathformer AMS，
    是否能够提升视频 memory 的多尺度时序建模能力。

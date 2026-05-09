# 电力领域结构化信息抽取 — 技术方案 v2

**项目代号**: AtlasNER v2
**任务**: 从电力风险管控文档中抽取结构化记录
**目标模型**: Qwen3.5-9B (generative SFT + Think)
**版本**: v2.0 | 2026-05-09

---

## 一、任务定义

### 1.1 输入与输出

```
输入: 电力领域文档段落（风险管控通知、规程、施工方案中的连续文本）
输出: 结构化风险管控记录（JSON 格式，含推理过程）
```

这是**生成式结构化抽取**，不是 token 级别 NER。模型阅读文档段落，经过推理（think），输出包含多个字段的 JSON 记录。

### 1.2 与 NER 的关键区别

| 维度 | Token-level NER | 本任务：结构化抽取 |
|------|----------------|---------------|
| 输出粒度 | 每 token 一个标签 | 每条记录 6 个字段，字段为完整文本 |
| 字段间关系 | 独立 span | 设备→工序→风险→措施 有强因果链 |
| 输出长度 | span 几个字 | 措施字段 median=185 字，max=2288 字 |
| 缺失处理 | 标 O | 模型需判断"该字段不存在"并跳过 |
| 模型架构 | 判别式 token classifier | 生成式 LLM + SFT |

### 1.3 模型选型

**Qwen3.5-9B**，生成式 SFT + LoRA 微调。

选型理由：
- 原生支持 `<think>` 模式，适合推理后输出结构化结果
- 9B 参数量在结构化抽取任务上有足够表达能力
- 中文电力领域术语覆盖优于同级英文模型
- LoRA + 4-bit 量化后可在消费级 GPU 上推理

硬件约束：
- 本地 16GB MPS 不足以训练 9B 模型（bf16 权重 ~18GB）
- **训练需使用云 GPU**（建议 A100 40GB 或 RTX 4090 24GB + QLoRA）
- 本地可做 4-bit 推理验证

---

## 二、统一输出 Schema

### 2.1 六个统一字段

从 9 个 Excel Sheet 的 22 个原始列归并而来：

| 字段 | 键名 | 归并来源 | 是否必需 |
|------|------|---------|---------|
| 设备 | `equipment` | 设备 / 设备类型 / 操作对象 | 否 |
| 工序 | `process` | 工序 / 操作行为 / 工作内容 | 否 |
| 风险描述 | `risk` | 主要风险点 / 基本风险点 / 存在的主要风险 / 风险类型 / 作业风险类型 / 风险可能导致的后果 | 否 |
| 风险等级 | `risk_level` | 风险等级 / 工序风险库等级 | 否 |
| 防范措施 | `prevention` | 风险防范措施 / 预控措施 / 对应防范措施 | 否 |
| 管控措施 | `control` | 工艺管控措施 / 质量管控措施 | 否 |

所有字段均为可选——模型仅输出文档中**确实存在**的字段。

### 2.2 字段值规范

- 字段值保留原文措辞，不改写不精简
- 措施字段保留编号列表格式（`1. ... 2. ...`）
- 风险描述如有多个，用中文顿号分隔
- 风险等级使用原始表述（"高"/"中"/"低" 或 "重大风险"/"较大风险"等）

### 2.3 关于 `risk` 字段的语义说明

`risk` 字段归并了两类语义不同的原始列：
- **风险类型** (如 "机械伤害、触电"): 风险事件的类别
- **风险后果** (如 "主变冲撞损坏、套管断裂"): 风险导致的具体后果

二者在当前 schema 中不做区分，均归入 `risk`。原因：
1. 原始 Excel 中两类表述经常混合使用
2. 业务上"风险点"和"后果"的边界模糊
3. 减少字段数有助于小模型学习

**评审关注点**: 如果业务要求区分风险类型和风险后果，需拆分为 `risk_type` 和 `risk_consequence` 两个字段。

---

## 三、数据层级与质量体系

### 3.1 三层数据架构

| 层级 | 代号 | 来源 | 用途 | 进入 dev/test |
|------|------|------|------|-------------|
| **Gold** | G0 | 人工审核确认的 Excel→SFT 转换 | 训练 + 验证 + 测试 | **是** |
| **Silver** | S1 | 旗舰模型标注真实文档 + 规则校验 + 抽样审核 | 仅训练 | **否** |
| **Synthetic** | S2 | 旗舰模型约束生成 + 自动校验 | 仅训练（补缺） | **否** |

### 3.2 训练时数据配比

| 阶段 | Gold | Silver | Synthetic |
|------|------|--------|-----------|
| Stage 1 预训练 | 50% | 35% | 15% |
| Stage 2 加入难例 | 40% | 35% | 25% |
| Stage 3 收敛微调 | 80% | 20% | 0% |

**最终收敛阶段必须以 Gold 数据为主**，防止模型过拟合 LLM 生成句式。

### 3.3 数据规模目标

| 数据类型 | 目标数量 | 来源 |
|---------|---------|------|
| Gold train | 1,500–1,800 | Excel 行转换 + 人工审核 |
| Gold dev | 150–200 | 从 Gold 中按文档分组抽出 |
| Gold test | 150–200 | 从 Gold 中按文档分组抽出 |
| Silver train | 2,000–5,000 | 45K 语料中旗舰模型标注 |
| Synthetic train | 1,000–3,000 | 旗舰模型约束生成 |
| **合计训练** | **~5,000–10,000** | — |

---

## 四、训练数据格式

### 4.1 SFT Conversation 格式

每条训练样本为一个 multi-turn conversation：

```json
{
  "id": "sgcc_000001",
  "messages": [
    {
      "role": "system",
      "content": "你是电力领域结构化信息抽取专家。给定电力文档段落，提取风险管控记录。仅输出文档中确实存在的字段，不臆造信息。输出严格JSON格式。\n\n可提取的字段：\n- equipment: 设备名称\n- process: 工序/作业内容\n- risk: 风险描述（风险类型或可能导致的后果）\n- risk_level: 风险等级\n- prevention: 防范措施\n- control: 管控措施"
    },
    {
      "role": "user",
      "content": "请从以下文档段落中提取结构化风险管控记录：\n\n<document>\n...\n</document>"
    },
    {
      "role": "assistant",
      "content": "<think>\n...\n</think>\n\n```json\n{\n  \"equipment\": \"...\",\n  ...\n}\n```"
    }
  ],
  "metadata": {
    "quality": "gold",
    "source_file": "89号文拆分标注.xlsx",
    "source_sheet": "Sheet1",
    "source_row": 2,
    "split": "train",
    "split_group": "89号文拆分标注_Sheet1_group_001",
    "field_count": 5,
    "input_char_count": 450,
    "output_char_count": 620
  }
}
```

### 4.2 输入构建策略

对于每条 Excel 行，需要构建一个"看起来像文档段落"的输入。三种策略：

**策略 A（Tier1 首选）: 文档风格重构**

将 Excel 字段重组为文档段落格式，模拟模型在推理时会看到的文档原文：

```
{设备} {工序}

主要风险：{风险描述}
风险等级：{风险等级}

风险防范措施：
{防范措施}

管控措施：
{管控措施}
```

**策略 B: 原始文本直接使用**

对于能从 `_matched_docs` 映射到原文的行，使用原文 PDF 中的实际段落（如果可获取）。

**策略 C（Synthetic）: 旗舰模型生成文档段落**

由旗舰模型生成风格多样的文档段落，包含各种格式变体（表格转文本、条款式、叙述式等）。

### 4.3 Think Chain 设计

Think chain 是本方案的核心训练信号，引导模型执行正确的抽取逻辑。

**Think Chain 模板结构**:

```
<think>
分析这段文档的结构和内容：

1. [设备识别] {判断过程} → equipment = "{值}" / 未找到设备信息
2. [工序识别] {判断过程} → process = "{值}" / 未找到工序信息
3. [风险识别] {判断过程} → risk = "{值}" / 未找到风险描述
4. [等级判断] {判断过程} → risk_level = "{值}" / 未标注风险等级
5. [防范措施] {定位过程} → prevention = "{条数}条措施" / 未找到防范措施
6. [管控措施] {定位过程} → control = "{条数}条措施" / 未找到管控措施
</think>
```

**设计原则**:
- 按固定字段顺序逐一推理，即使某字段不存在
- 对存在的字段：引用文档中的关键词/句作为定位依据
- 对不存在的字段：明确标注"文档中未找到该信息，跳过"
- 对措施字段：标注条目数量，帮助模型学习完整性
- Think 长度控制在 80-200 字（简单记录短，复杂记录长）
- **不使用过长的 chain-of-thought**——think 是提取引导，不是分析论文

### 4.4 不同复杂度的 Think 示例

**完整记录（5-7 字段）**:
```
<think>
分析文档内容：
1. 设备：标题明确指出"变压器"。→ equipment = "变压器"
2. 工序："变压器整装转运"，出现在标题中。→ process = "变压器整装转运"
3. 风险："机械伤害、物体打击"。→ risk = "机械伤害、物体打击"
4. 风险等级：文档未标注等级。→ 跳过
5. 防范措施：共3条编号措施，从"施工前确认"到"采用防护措施"。→ 完整提取
6. 管控措施：共4条编号措施，从"移动轨迹"到"单次顶推控制在80cm内"。→ 完整提取
</think>
```

**简单记录（2 字段）**:
```
<think>
文档内容较简短，仅包含风险点和对应措施。
1. 设备/工序/等级：均未提及。→ 跳过
2. 风险："机械伤害"。→ risk = "机械伤害"
3. 防范措施：1条，"施工前确认顶升及顶推位置地面承受力。" → 完整提取
</think>
```

---

## 五、数据构建流程

### 阶段 0: 冻结标注规范（本文档 + annotation_guideline.md）

**产出**:
- `docs/annotation_guideline.md` — 字段定义、边界规则、冲突处理
- `data/schema/unified_schema.json` — 机器可读 schema
- 本文档

### 阶段 1: Gold 数据构建

**输入**: `labeled_excel_rows.json` (1,852 行)

**流程**:
```
1,852 行 Excel
    ↓ 字段归并到统一 schema
    ↓ 过滤只含序号/无实质内容的行
    ↓ 构建文档风格输入
    ↓ 生成 think chain
    ↓ 格式化为 SFT conversation
    ↓ 按 source_file + sheet 分组
    ↓ 分组级别切分 train/dev/test (80/10/10)
    ↓
Gold train (~1,500) + Gold dev (~175) + Gold test (~175)
```

**关键规则**:
- **先切分，后增强**: dev/test 确定后锁定，任何增强只作用于 train
- **按文档分组切分**: 同一 source_file + sheet 的行必须在同一 split
- **近重复检测**: 同一 Excel 行在不同 Sheet 可能重复出现，必须归入同一 split

### 阶段 2: Silver 数据标注

**输入**: `cleaned_corpus.jsonl` (45,296 句)

**流程**:
```
45K 语料
    ↓ 规则预筛选（含专业术语/标准号/风险词汇的句子）
    ↓ 聚合为段落级别（同一 source_row 的句子合并）
    ↓ 旗舰模型标注（生成 think + JSON）
    ↓ 规则校验（JSON 格式、字段值非空、措施条目计数）
    ↓ 第二模型审核（独立标注，对比差异）
    ↓ 一致的进入 Silver，不一致的进入人工审核池
    ↓
Silver train (2,000-5,000)
```

### 阶段 3: Synthetic 数据生成

**目的**: 补齐稀缺场景，不是追求数量。

**生成目标**:

| 缺口 | 合成策略 | 目标量 |
|------|---------|--------|
| 少字段记录（1-2 字段） | 生成只含 risk + prevention 的简单记录 | ~300 |
| 无实体文档（负例） | 生成纯管理要求段落，正确输出为空 JSON | ~200 |
| 长措施文档 | 生成含 5+ 条措施的复杂段落 | ~200 |
| 多记录文档 | 一个段落含多条记录，输出 JSON array | ~150 |
| 格式变体 | 表格转文本、条款式、通知式等不同格式 | ~300 |
| 直流/输电/配电跨专业 | 覆盖训练集中较少的专业 | ~300 |

### 阶段 4: 质量审核与最终集成

- Gold: 全量人工审核 think chain 和 JSON 正确性
- Silver: 抽样 20% 人工审核
- Synthetic: 规则校验 100% + 抽样 10% 人工审核
- 验证 dev/test 无数据泄露（无近重复进入不同 split）

---

## 六、训练方案

### 6.1 模型配置

```yaml
model: Qwen/Qwen3.5-9B
method: SFT (Supervised Fine-Tuning)
adapter: LoRA
  rank: 32
  alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  dropout: 0.05
quantization: QLoRA 4-bit (if GPU VRAM < 40GB)
dtype: bfloat16

max_length: 2048
batch_size: 2 (per GPU)
grad_accum_steps: 8
effective_batch_size: 16
learning_rate: 1.0e-4
warmup_ratio: 0.05
epochs: 3-5
optimizer: AdamW
scheduler: cosine
gradient_checkpointing: true
```

### 6.2 训练阶段

| 阶段 | 数据 | Epochs | 目标 |
|------|------|--------|------|
| Stage 1 | Gold + Silver + Synthetic (按 50:35:15 采样) | 2 | 学习基本抽取能力 |
| Stage 2 | + 难例数据（错误分析后补充） | 1 | 提升边界情况处理 |
| Stage 3 | Gold only (或 Gold 80% + Silver 20%) | 1-2 | 收敛到真实分布 |

### 6.3 Loss 计算

- 仅对 `assistant` 回复部分计算 loss（system + user 部分 mask 为 -100）
- Think 部分和 JSON 部分均参与 loss
- 不使用 token-level class weighting（生成式任务不需要）

---

## 七、评估方案

### 7.1 字段级指标

| 指标 | 计算方式 | 适用字段 |
|------|---------|---------|
| **Exact Match (EM)** | 归一化后完全一致 | equipment, process, risk_level |
| **Fuzzy Match (FM)** | 归一化编辑距离 ≥ 0.85 | risk |
| **ROUGE-L** | 最长公共子序列 | prevention, control |
| **Field Recall** | 应输出的字段是否输出了 | 所有字段 |
| **Field Precision** | 输出的字段是否应该输出 | 所有字段 |
| **Record F1** | 基于 field-level P/R 的 micro F1 | 整条记录 |

归一化规则：去除首尾空白、统一全半角、统一换行符。

### 7.2 分字段分析

每个字段单独报告 EM/FM/ROUGE-L，识别薄弱字段。

### 7.3 JSON 格式合规率

```
valid_json_rate = 输出可解析为合法JSON的样本数 / 总样本数
```

目标：≥ 98%。

### 7.4 OOD 测试

从未参与训练/增强的补充文档中选取段落，构建独立 OOD 测试集（~50 条人工标注），评估泛化能力。

### 7.5 必做 Ablation

| 实验 | 目的 |
|------|------|
| Gold only | 基线 |
| Gold + Silver | Silver 数据是否有效 |
| Gold + Synthetic | Synthetic 数据是否有效 |
| 有 think vs 无 think | Think chain 是否提升抽取质量 |
| Stage 3 收敛 vs 不收敛 | Gold-heavy 微调是否有效 |

---

## 八、执行路线图

| 阶段 | 工作 | 产出 | 预计耗时 |
|------|------|------|---------|
| **P0** | 冻结 schema + 标注规范 | 本文档 + guideline | 当前完成 |
| **P1** | Excel→Gold SFT 转换 | gold_train/dev/test.jsonl | 1-2 天 |
| **P2** | Synthetic 生成 | synthetic_train.jsonl | 1-2 天 |
| **P3** | Silver 标注（45K 语料） | silver_train.jsonl | 2-3 天 |
| **P4** | 人工审核 Gold + 抽样 Silver | 审核报告 | 3-5 天 |
| **P5** | 训练 Stage 1-3 | 模型 checkpoint | 1-2 天 (GPU) |
| **P6** | 评估 + 错误分析 | eval 报告 | 1 天 |
| **P7** | 补数据 + 迭代 | 更新数据集 | 持续 |

---

## 九、交付物清单

```
docs/
  technical_plan_v2.md          # 本文档
  annotation_guideline.md       # 标注规范
data/
  schema/
    unified_schema.json         # 机器可读 schema
  gold/
    train.jsonl                 # Gold 训练集
    dev.jsonl                   # Gold 验证集
    test.jsonl                  # Gold 测试集
  silver/
    train.jsonl                 # Silver 训练集
  synthetic/
    train.jsonl                 # Synthetic 训练集
  reports/
    data_statistics.json        # 数据统计
    split_manifest.json         # 切分清单
    quality_report.md           # 质量报告
scripts/
  build_sft_dataset.py          # Gold 数据构建脚本
  generate_synthetic.py         # Synthetic 数据生成脚本
  evaluate.py                   # 评估脚本
```

---

## 十、评审关注点

1. **统一 schema 的 `risk` 字段**归并了"风险类型"和"风险后果"，是否需要拆分？
2. **Qwen3.5-9B** 在 SFT 后输出长文本 JSON 的可靠性？是否需要 constrained decoding？
3. **Think chain 生成质量**：由旗舰模型自动生成的 think chain 能否有效引导 9B 模型？
4. **Silver 数据**从 45K 语料标注，但语料本身是从 Excel 分句而来（非原始文档），是否有信息泄露风险？
5. **负例训练**比例是否充足？纯管理段落（不含风险记录）在推理时出现频率可能很高。
6. **多记录抽取**（一个段落含多条记录）的训练数据是否足够？

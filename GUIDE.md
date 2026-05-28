# 小学数学应用题自动解题 — 实操示范总结

## 本地验证结果

在本地 CPU 上用小样本验证了 Qwen2.5-0.5B 的推理和训练流程：

### 零样本 CoT（方案1，不需要训练）

用"请逐步推理，最后以答案是[数字]格式给出答案"的提示词，在 5 道测试题上跑 CoT 推理：

| 题目 | 期望答案 | 模型输出 | 结果 |
|------|----------|----------|------|
| (91.64+7.36)×(43.6-3.6) | 3960 | 3960 | 正确 |
| 书架上下层问题 | 18 | 91 | 错误 |
| 32人÷3人/船 | 11 | 11 | 正确 |
| 课堂时间计算 | 0.25 | 23.33 | 错误 |
| 50元÷7.9元/盒 | 6 | 6 | 正确 |

**准确率 3/5 = 60%**，而直接回答模式（不要求逐步推理）准确率 0/5。说明 CoT 提示词对 0.5B 小模型有明显提升。

### SFT 训练（方案2，验证 pipeline）

用 30 条训练数据、1 epoch、CPU、LoRA (r=8)，训练耗时 **1分44秒**。pipeline 正常运行，loss 下降（3.36）。30 条数据太少导致 SFT 效果不明显，但这只是验证——在 GPU 上跑完整 12000 条数据 + 3~5 epoch 才会看到实质提升。

---

## 远程服务器操作指南

### 环境准备

```bash
# 1. 克隆项目（如果你是直接拷贝代码，跳过这一步）
git clone <your-repo> math_solver && cd math_solver

# 2. 创建 conda 环境
conda create -n math_solver python=3.10 -y
conda activate math_solver

# 3. 安装依赖
pip install -r requirements.txt
```

### 下载模型

```bash
python -c "from modelscope import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./', revision='master')"
```

下载后把模型路径修正为 `./Qwen/Qwen2.5-0.5B-Instruct/`（modelscope 可能把文件放在带下划线的缓存目录，需要手动处理）。

### 方案1：CoT 提示词推理（不训练，直接跑）

```bash
# 零样本 CoT
python cot_prompting.py --mode zero_shot
# 产出: submit_cot_zero_shot.csv

# 少样本 CoT
python cot_prompting.py --mode few_shot
# 产出: submit_cot_few_shot.csv

# 建议先小样本测试
python cot_prompting.py --mode zero_shot --max_samples 10
```

### 方案2：数据构建 + CoT SFT 训练

```bash
# Step A: 构建 CoT 数据（给训练集补充推理步骤）
# 建议先用 500 条测试，确认没问题后跑完整 12000 条
python build_cot_data.py --max_samples 500
python build_cot_data.py --augment    # 跑完整数据 + 数据增强

# Step B: SFT 训练
python sft_cot_train.py --num_epochs 5 --batch_size 4
# 完整训练约 2-4 小时 (GPU)

# Step C: 推理
python infer_all.py --mode sft --lora_path ./output/Qwen_COT
# 产出: submit_sft.csv

# Step D: 本地评估（用 train.json 验证）
python evaluate.py submit_sft.csv
```

### 方案3：DPO 训练（需要先修好 trl 包）

```bash
# 包问题：trl 1.5.0 与 transformers 5.9.0 不兼容
# 如果遇到 trl 导入错误，运行：
pip install trl==0.12.0

# Step A: 构建 DPO 偏好数据
python build_dpo_data.py

# Step B: DPO 训练
python dpo_train.py --num_epochs 1 --batch_size 4

# Step C: 推理
python infer_all.py --mode dpo --lora_path ./output/Qwen_DPO
# 产出: submit_dpo.csv
```

### 提交到比赛平台

把生成的 `submit_xxx.csv` 提交到 https://www.datafountain.cn/competitions/467 的"作品提交"入口。每日 3 次提交机会。

---

## 文件说明

### 数据文件

| 文件 | 作用 |
|------|------|
| `train.json` | 训练集，12000 条。每题有 `question`（题目）、`answer`（正确答案）、`instruction`（要求直接输出数字答案）。**你可以修改此文件来构建增强数据**。 |
| `test.json` | 测试集，8000 条。每题有 `question`、无答案。**规则严禁处理测试数据**。 |
| `submit.csv` | Baseline 的提交模板，文件格式为 `id,答案`，无表头。每次推理产出同格式的 csv 文件提交到平台。 |

### Baseline 文件（原始仓库自带）

| 文件 | 作用 |
|------|------|
| `qwen_ft.py` | Baseline 训练脚本。直接对 train.json 做 SFT，让模型学会「读题 → 输出答案」。不涉及 CoT 推理。 |
| `infer.py` | Baseline 推理脚本。加载训练好的 LoRA 模型，对 test.json 逐题推理，生成 submit.csv。 |
| `README.md` | 原始仓库的简要说明。 |

### 公用模块

| 文件 | 作用 |
|------|------|
| `utils.py` | 所有脚本共用的工具函数：`extract_answer()` 从模型输出中提取数字答案、`load_json()` / `save_json()`、`load_csv_submit()` / `save_csv_submit()` 读写提交文件、`evaluate()` 本地评估准确率、`FEW_SHOT_EXAMPLES` 少样本 CoT 的示例题库。 |

### 方案1：CoT 提示词（不训练）

| 文件 | 作用 |
|------|------|
| `cot_prompting.py` | 加载 Qwen-0.5B 基础模型（不经任何训练），用 CoT 提示词直接做推理。支持 `--mode zero_shot`（提示词含"请逐步推理"）和 `--mode few_shot`（提示词含 4 道带推理步骤的示例题）。产出 `submit_cot_{mode}.csv`。 |

### 方案2：数据构建 + CoT SFT 训练

| 文件 | 作用 |
|------|------|
| `build_cot_data.py` | **Phase 1**：给 train.json 的每道题生成推理步骤。用 base 模型 + 正确答案作为提示，让模型「反向」补出解题过程。可选 `--augment` 做数据增强（修改原题中的数字，乘 2/3/0.5 生成新题）。产出 `train_cot.json`，格式：`{question, answer, reasoning, instruction}`。 |
| `sft_cot_train.py` | **Phase 2**：用 train_cot.json（或其他含 `reasoning` 字段的数据）做 LoRA SFT 训练。让模型学会「读题 → 逐步推理 → 输出答案」。核心参数：`--num_epochs`、`--batch_size`、`--device auto/cpu`。产出 `./output/Qwen_COT/` 检查点。 |

### 方案3：DPO 训练

| 文件 | 作用 |
|------|------|
| `build_dpo_data.py` | **Phase 1**：构建偏好对数据。对每道题生成两种推理：**chosen**（给定正确答案 → 产生正确推理）和 **rejected**（给定错误答案或直接推理 → 产生可能错误的推理）。产出 `train_dpo.json`，格式：`{question, instruction, chosen, rejected}`。 |
| `dpo_train.py` | **Phase 2**：用 train_dpo.json 做 DPO 训练（基于 TRL 库的 `DPOTrainer`）。让模型学会区分和偏好正确的推理链。核心参数：`--num_epochs`、`--batch_size`、`--beta`（DPO 温度）。产出 `./output/Qwen_DPO/` 检查点。 |

### 推理与评估

| 文件 | 作用 |
|------|------|
| `infer_all.py` | 统一推理入口。支持 `--mode base/sft/dpo`，对应三种模型来源（基础模型 / SFT LoRA / DPO LoRA）。加载模型后对 test.json 做推理，自动提取答案，产出 submit csv。 |
| `evaluate.py` | 本地评估脚本。给定一个 submit csv 文件，用 train.json 的标签计算准确率。用法：`python evaluate.py submit_sft.csv`。**注意**：test.json 没有标签，无法直接用此脚本评估——必须提交到比赛平台看分数。 |

### 配置文件

| 文件 | 作用 |
|------|------|
| `requirements.txt` | Python 依赖清单：torch、transformers、modelscope、peft、trl、accelerate、tqdm。 |
| `.gitignore` | 忽略模型文件（`Qwen/`）、训练产出（`output/`）、生成的 csv 和 json、缓存目录。 |

### 文件之间的依赖关系

```
train.json ──────────────────────────────────────────────┐
                                                        │
方案1（不训练）:                                          │
  cot_prompting.py ───→ submit_cot_{mode}.csv            │
                                                        │
方案2（训练）:                                            │
  train.json ──→ build_cot_data.py ──→ train_cot.json    │
                                         │               │
                                         ▼               │
                                   sft_cot_train.py      │
                                         │               │
                           ./output/Qwen_COT/            │
                                         │               │
                                         ▼               │
                                   infer_all.py ──→ submit_sft.csv
                                                        │
方案3（训练）:                                            │
  train.json ──→ build_dpo_data.py ──→ train_dpo.json    │
                                         │               │
                                         ▼               │
                                    dpo_train.py         │
                                         │               │
                           ./output/Qwen_DPO/            │
                                         │               │
                                         ▼               │
                                   infer_all.py ──→ submit_dpo.csv

评估:
  submit_*.csv ──→ evaluate.py ──→ 准确率（对照 train.json）
  submit_*.csv ──→ 提交到比赛平台 ──→ 平台得分
```

---

## 按课程要求你需要做的事

### 比赛截止：2026年6月11日

| 事项 | 说明 |
|------|------|
| **提交 csv 到平台** | 用方案1/2/3 各自生成 csv，挑效果最好的提交。提交到 DataFountain 平台后截图保存成绩。 |
| **截图成绩** | 平台上的最好排名和分数截图，上传到 elearning。 |
| **CSV 文件** | 提交最好的 csv 文件到 elearning。 |

### 报告截止：2026年6月19日

| 事项 | 说明 |
|------|------|
| **最终代码** | 上传所有代码到 elearning。 |
| **方案报告（4 页）** | 报告应包含：每个方案的实现描述、实验结果（准确率对比）、实验分析（为什么某个方案好/不好）。 |
| **PPT**（可选） | 如果有 15 周汇报，准备 PPT。 |

### 计分公式

```
比赛分 s1 = score × 15    （score 是平台准确率）
工作量分 s2 = 方案数 × 3  （组队，每个方案 3 分；个人的话每个 5 分）
总分 = min(s1 + s2, 15)
```

你目前有 3 个方案（CoT提示、数据构建+SFT、DPO），工作量分 = 9 分。只要平台 score > 0.4，总分就满 15 分。

如果你打算再做第 4 个方案（GRPO），就是 12 分工作量分，几乎不需要比赛分数就能满。

### 推荐顺序

1. **先提交方案1**（CoT 零样本 + 少样本）→ 最快出分
2. **跑方案2**（数据构建 + SFT）→ 在 GPU 上跑完整训练
3. **跑方案3**（DPO）→ 依赖方案2的 CoT 数据
4. **写报告**（4 页）→ 分析三种方案的效果差异

# AIC 2026 无人机低空图像语义分割项目 —— 代码结构与训练流程分析报告

> 分析对象：`https://github.com/passing666/aic`（main 分支，commit `df86d3b`）
> 分析方法：GitHub 全量文件阅读（README、src/ 全部 6 个脚本、docs/、outputs/results/ 关键结果）。
> 限制：正式数据、CSV 划分、预训练权重、运行日志未上传公开仓库，因此本报告基于静态代码阅读，未在本机实跑训练。

---

## 1. 项目概况

| 项目 | 内容 |
|---|---|
| 赛题 | AIC 2026 无人机低空航拍图像语义分割 |
| 任务 | 输入 RGB 航拍图（官方 1024×1024），输出同尺寸单通道类别 ID 图 |
| 类别 | 0~8 共 9 类；0 = Ignore 不参与 mIoU 与 loss，有效评价类别 1~8 |
| 数据集 | 正式训练集 6996 张图 + 6996 张 mask，均 1024×1024 |
| 固定划分 | 训练 5597 / 验证 1399，随机种子 2026 |
| 模型 | Hugging Face SegFormer MIT-B0（约 371 万参数，全量微调） |
| 输入尺寸 | 基线 512×512；exp002 提升至 768×768 |
| 优化 | AdamW（lr=6e-5, wd=0.01），CrossEntropy（ignore_index=0），AMP |
| 增强 | 仅同步水平翻转 + ImageNet 归一化 |
| 当前最佳 | exp002：验证 mIoU **0.76258**（epoch 30），较 exp001 正式基线 0.74939 提升 +0.01320 |

官方定位（2026-09-05 官网核验）：本赛题为《第八届全球校园人工智能算法精英大赛》（AIC）**算法挑战赛道**赛题之一，赛道实行"赛马制"——全国统一组织、不分赛区，复赛等同省级赛事；赛程为**初赛 → 复赛 → 半决赛 → 总决赛**四阶段。官网逐条核对见附录 A。

进度要点：数据体检通过 → 固定划分生成并审计 → exp001（512 基线）→ 评价代码修复（Ignore 掩码）→ exp002（768）成为当前最佳。团队按"一次只改一个变量"的原则推进实验。

## 2. 仓库结构总览

```text
aic/
├── README.md                 项目说明、类别表、实验规范、上传边界
├── requirements.txt          numpy / Pillow / torch / transformers（无版本 pin）
├── .gitignore                数据、权重、runs、本地报告等不提交
├── src/                      可运行代码
│   ├── check_dataset.py      数据体检：尺寸/配对/损坏/标签值/像素直方图
│   ├── make_split.py         生成固定 train/val 划分 + 类别分布摘要
│   ├── audit_split.py        划分审计：重复 ID、跨集合哈希、损坏与分布
│   ├── train.py              SegFormer 训练/验证/逐 epoch 记录（核心）
│   ├── evaluate_checkpoint.py 用修复后的 Ignore 掩码重评已存 checkpoint
│   └── smoke_test.py         环境 + 单样本 + 小模型链路冒烟
├── docs/analysis/expXXX/     实验事实分析（exp001_baseline.md、exp002_size768.md）
├── docs/suggestions/expXXX/  团队建议与决策记录
└── outputs/
    ├── audits/split_audit_report.json   划分审计报告（约 1.1 MB）
    └── results/
        ├── exp001_baseline/             history.json（旧口径）、exp001_fix_metrics.json
        └── exp002_size768/              run_config/history.jsonl/history.json/
                                         best_metrics.json/summary.json
```

各脚本职责与依赖关系：

| 脚本 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `check_dataset.py` | images/、masks/、Label.txt | 控制台摘要 + JSON 报告 | 只读校验，不改数据 |
| `make_split.py` | 同上 | train.csv / val.csv / split_report.json | 固定种子随机划分 |
| `audit_split.py` | data-root + train/val.csv | 审计 JSON | ID/哈希/损坏/分布审计 |
| `train.py` | data-root + train/val.csv + mit-b0 权重 | 实验目录全套产物 | 训练 + 验证 + 持久化 |
| `evaluate_checkpoint.py` | data-root + val.csv + checkpoint | 重评 JSON | 修复口径的独立评价 |
| `smoke_test.py` | images/、masks/ | 控制台 | 冒烟测试 |

> 注意：`.gitignore` 忽略了 `outputs/splits/*.csv`、`best_model/`、`*.pt` 等，因此仓库内看不到划分 CSV 与权重，分析只能到"读取 id 列、由 data_root 拼路径"这一层。

## 3. 数据准备链路

**① `check_dataset.py` —— 数据体检**
- 对比 images/ 与 masks/ 同名配对，报告缺失、损坏（OSError/UnidentifiedImageError）、尺寸不匹配。
- 读取每个 mask 的像素值集合，按 Label.txt（或默认 0~8）标出非法标签；统计各类出现图数/像素直方图。
- `status` 只在全部检查通过时置 `"PASS"`（main 返回码 0/1）。README 记录正式训练集 6996/6996 全部 PASS。

**② `make_split.py` —— 固定划分**
- `random.Random(seed).shuffle(pairs)` 后取前 `round(n * val_ratio)` 为验证集（6996×0.2=1399.2 → 1399，与 README 5597/1399 一致）。
- 写出 `train.csv` / `val.csv`（列：`id, image, mask`），并生成两侧的类别图数/像素分布摘要。
- CSV 中 `image`/`mask` 列保存的是**本机绝对路径**（Windows 或 AutoDL 路径）。`train.py` 只读 `id` 列再拼 `--data-root`，因此同一份 CSV 可跨平台使用——这是刻意的兼容设计。

**③ `audit_split.py` —— 划分审计**
- 检查 train/val 是否有重复 ID、各集合内部是否有重复 ID。
- 对样本做 SHA-256（`--hash-limit 0` 全量），比较跨集合图片/mask 哈希，探测精确重复与泄漏。
- 报告尺寸错误、损坏与训练/验证类别像素比例。
- 已确认结论（README + docs）：无重复 ID、无跨集合重复图片、唯一 1 对重复 mask（train `2018` 与 val `4470`，图片不同）；尚未做感知哈希/特征级近重复检查。

**④ `smoke_test.py` —— 链路冒烟**
- 打印 torch/torch CUDA/CUDA 可用性/GPU 名；加载 1 张图 + 同名 mask，检查标签值 0~8；用随机初始化的 1×1 Conv2d(3→9) 跑一次前向 + CE loss。

## 4. 训练主流程（`train.py`）

整体流程：

```text
原始数据(1024×1024) ──check_dataset──▶ 体检通过
      │
      ▼
make_split(seed=2026) ──▶ train.csv(5597) / val.csv(1399) + 分布摘要
      │
      ▼
audit_split ──▶ 审计报告（无泄漏/损坏，1 对重复 mask 不阻塞）
      │
      ▼
train.py
  ① 固定随机种子(2026) ② 构建 Dataset/DataLoader
  ③ 载入 mit-b0(SegformerForSemanticSegmentation, num_labels=9, ignore_mismatched_sizes)
  ④ AdamW(lr=6e-5, wd=0.01) + GradScaler(AMP)
  ⑤ 每 epoch：训练步(autocast → 上采样 logits → CE ignore=0) → 验证 mIoU
     → 写 history.jsonl(逐行 flush) + history.json(全量快照)
     → 若 val_miou 创新高：save_pretrained(best_model) + best_metrics.*
  ⑥ 结束：summary.json（best_epoch/best_miou/耗时）
      │
      ▼
outputs/results/expXXX/ ──evaluate_checkpoint(修复口径)──▶ 正式基线
```

**step-by-step（附行号）**

1. **参数与设备（L136–155）**：`--data-root / --train-csv / --val-csv / --output-dir / --image-size(512) / --batch-size(2) / --epochs(20) / --learning-rate(6e-5) / --num-workers(2) / --device(auto) / --no-amp / --pretrained(nvidia/mit-b0) / --log-file`。`device=auto` 时 CUDA 可用即用 cuda；显式 cuda 而不可用则报错。
2. **日志与随机性（L156–166）**：`--log-file` 用 `Tee` 把 stdout/stderr 同时写终端和文件；固定 `random/np.random/torch.manual_seed(2026)`。
3. **数据加载（L63–98, L168–171）**：`SegmentationDataset` 从 CSV 读 `id` 列；`__getitem__` 内**实时**打开 PNG——图转 RGB、训练时 50% 水平翻转、BILINEAR resize 到 image_size、(x/255 − mean)/std；mask 转 L、**同步翻转**、NEAREST resize、校验像素 ∈[0,8]。train loader `shuffle=True`，val `shuffle=False`，均 `pin_memory`。
4. **模型（L173–176）**：`SegformerConfig.from_pretrained(..., num_labels=9, semantic_loss_ignore_index=0)` + `from_pretrained(..., ignore_mismatched_sizes=True)`，分割头重新初始化，主干为 ImageNet 预训练 MIT-B0。
5. **优化器与 AMP（L198–199）**：`AdamW(lr, weight_decay=0.01)`；`GradScaler("cuda")`，仅 cuda 且未加 `--no-amp` 时启用。
6. **单 epoch 训练（L207–220）**：每 batch：`zero_grad(set_to_none=True)` → `autocast` 内前向，SegFormer 输出（1/4 分辨率）`interpolate` 回 mask 尺寸 → `cross_entropy(ignore_index=0)` → `scaler.scale(loss).backward() / scaler.step / scaler.update`；累加 `train_loss`。
7. **验证与记录（L221–238）**：每 epoch 结束跑一遍完整验证集得到 mIoU 与 8 类 IoU；record 含 `epoch/train_loss/val_miou/ious/elapsed_seconds`；`history.jsonl` 每行一条并立即 flush，`history.json` 全量重写；mIoU 创新高时保存 `best_model/`（HF 格式）、`best_metrics.pt`、`best_metrics.json`。
8. **收尾（L239–256）**：写 `summary.json`（best_epoch、best_miou、final_miou、elapsed_seconds/minutes），恢复 stdout 并关闭日志。

**验证 / 评价链路**

- 训练内评价 `evaluate()`（L115–132）：跨全验证集累计各类 intersection/union。当前代码已带 `valid = masks != 0` 掩码（修复后的口径）。
- 独立重评 `evaluate_checkpoint.py`：`from train import ...` 复用 Dataset，载入已存 checkpoint 跑验证集，额外统计 `predicted_ignore_ratio`（有效 GT 像素中被预测为类别 0 的比例）。exp001 的 epoch-25 权重即用此脚本重评得到 0.74939。
- 评价口径要点：类别 0 不参与 mIoU；GT=0 的像素被 `valid` 排除，不会进入任何类别的并集。见第 6 节对其边界的讨论。

## 5. 实验结果现状（以仓库内 JSON 为准）

exp001（512×512 基线，旧口径 30 个 epoch 记录）+ 修复口径重评：

| 指标 | 值 |
|---|---:|
| 旧口径最佳 epoch / mIoU | 25 / 0.74514 |
| 修复口径正式基线 mIoU（epoch 25 重评） | **0.74939** |
| 修复前后差 | +0.00425 |
| 有效像素预测为 Ignore 比例 | 0.00000 |

exp002（768×768，`run_config.json` / `best_metrics.json` / `summary.json`）：

| 指标 | 值 |
|---|---:|
| 最佳 epoch | 30（= 末轮，未见平台期） |
| 验证 mIoU | **0.76258** |
| 相对 exp001 正式基线 | +0.01320 |
| 训练总耗时 | 约 101.1 分钟（6066.7s） |
| 配置 | image_size 768, batch 2, lr 6e-5, AMP on, 参数 3,716,457 |

每类 IoU 对比（最佳轮次）：

| ID | 类别 | exp001(512) | exp002(768) | 变化 |
|---:|---|---:|---:|---:|
| 1 | Background | 0.68018 | 0.68859 | +0.00840 |
| 2 | Building | 0.80444 | 0.83341 | +0.02897 |
| 3 | Road | 0.78765 | 0.80547 | +0.01782 |
| 4 | Water | 0.85413 | 0.86859 | +0.01447 |
| 5 | Barren | 0.53582 | 0.54365 | +0.00783 |
| 6 | Vegetation | 0.86888 | 0.87615 | +0.00727 |
| 7 | Agricultural | 0.74676 | 0.75743 | +0.01066 |
| 8 | Vehicle | 0.71723 | 0.72738 | +0.01016 |

要点：8 个有效类别全部提升；**Barren（最低）、Vehicle、Background** 仍是短板；exp002 最佳=末轮说明 30 epoch 可能未收敛完。

## 6. 潜在问题分析

按严重程度 P0（风险高，建议优先处理）→ P1（改进空间）→ P2（低风险/工程整洁，见下文）。

### P0 级

**P0-1：没有真正的中断续训机制，一次训练约 100 分钟、中断即前功尽弃**
- 位置：`train.py` L198–239。主循环既无 checkpoint 加载（restore epoch/optimizer/scaler/RNG），也不保存"最近权重"。
- 现状：`history.jsonl/history.json` 保存的是**已完成的记录**（README 表述为"恢复已完成记录"，正确），`best_model/` 只保留历史最优权重；若训练在 epoch 20/30 中断，本轮权重与优化器状态全部丢失，只能从头重跑。
- 建议：定期保存 `last.pt`（model/optimizer/scaler/RNG/epoch）并支持 `--resume`；或至少额外保存一份 `latest_model/`。

**P0-2：缺少推理/提交链路**
- 现状：`src/` 没有对官方 `test_1` 的推理脚本，也没有"输出必须 1024×1024、单通道、非 Palette、像素 ID 合法"的提交自检器。docs（`exp001_team_suggestions.md` 的 P1 项）与 README 均要求此类产物，但仓库尚未实现。
- 风险：验证跑得再好，正式提交仍需从 checkpoint 复原 + 后处理；手工推理易在尺寸、通道、ID 映射上出错。
- 建议：尽快补 `predict.py`（加载 best_model → 1024 推理 → 保存同名 PNG）与提交校验器；这也是 exp002 最佳模型产生价值的唯一通道。

> 勘误注：原 P0-3（评价口径对"有效像素预测为 Ignore"的处理）的机制分析有误，已勘误并降级迁移到 P1 节 **P1-8**，此处不再展开。

### P1 级

**P1-1：无学习率调度/热身，且 exp002 末轮仍在上涨**
- 位置：`train.py` L198 之后、主循环之前无任何 `lr_scheduler` 配置。SegFormer 官方微调常用 warmup + poly；exp002 `best_epoch=30` 说明 30 epoch 未到平台期。
- 建议：作为下一个单变量实验（团队路线中的 exp003）比较 `cosine/poly + warmup` 或延长 epoch。

**P1-2：数据增强极弱，且不含颜色/尺度变化**
- 位置：`SegmentationDataset.__getitem__` L80–92，仅 50% 水平翻转 + ImageNet 归一化。
- 影响：低空航拍同域性强，但小目标（车辆/道路边缘）与类别不平衡更需要增强与正则；docs 已把"轻度颜色增强""CE+Dice""温和类别权重"列为候选，须逐个引入以保持控制变量。

**P1-3：类别不平衡无任何处理**
- Barren（0.536）、Vehicle（0.727）、Background（0.680）显著低于其余类别；CE 无类别权重、无 Dice/辅助 loss。类别像素比例数据在 `outputs/audits/split_audit_report.json`（约 1.1 MB，未随报告入库），建议据此设计采样或加权。

**P1-4：评价分辨率与官方提交分辨率不一致**
- 训练/验证分别在 512/768 下进行（mask 用 NEAREST 缩放），而官方按 1024 GT 评提交。低分辨率下 GT 边界略劣化，且与最终 1024 推理存在口径差。
- 建议：`evaluate_checkpoint.py` 已支持任意 `--image-size`；最终选型前增加一次 1024 验证集重评并记为正式基线。

**P1-5：best 模型只按单次验证 mIoU 挑选，验证集仅 1399 张**
- 位置：L234–238。单次 val 存在小样本噪声；可对验证做翻转 TTA/多次平均后再选，或对权重做 EMA。低风险改进。

**P1-6：输出目录无覆盖保护**
- 位置：L156 `args.output_dir.mkdir(...)`；若目录已含 `history.json` 仍会继续训练，L204 以 `"w"` 打开 JSONL 直接清空旧记录。
- 建议：检测到 `summary.json`/`history.json` 已存在且参数不一致时拒绝运行，或要求显式 `--overwrite`。当前依赖团队纪律防覆盖。

**P1-7：CPU 侧数据管线可能成为瓶颈**
- 位置：L63–98（每取一个样本即时解码 + resize + 归一化）、L145 `num_workers=2` 且无 `persistent_workers`。5597 张每 epoch 全量重复解码，768 输入下开销更大；若 GPU 利用率不足会直接拖慢单 epoch。
- 建议：调大 `num_workers`、启用 `persistent_workers`/`prefetch_factor`；如资源允许可缓存增强结果或对比更快的解码库。

**P1-8：缺少对"有效像素被预测为 Ignore"的观测（原 P0-3，经勘误降级）**
- 勘误：旧稿称"有效像素被判为 0 时既不进 TP 也不进 FP、mIoU 会被系统性高估"，该结论不成立。按 `evaluate()`（L124–129）与 `compute_miou()`（L104–110）的实现，GT=c 且预测=0 的像素经 `actual = (targets == c) & valid` 仍计入类别 c 的 union，但不进 intersection，等价于类别 c 的一个 FN，会照常降低 IoU_c——预测成 0 并不"隐形"，mIoU 也不会因此被高估（详见附录 A 勘误记录）。
- 残余风险：把难像素输出为 0 只伤害其真值类（FN）；若错成另一个真实类别 d，则同时伤害真值类（FN）与被猜类（FP）。因此 0 通道是"更温和的错误出口"，模型在训练不足或难像素集中时可能学会依赖它来"逃避"。
- 现状缺口收敛为观测缺失：训练 record（L222–228）不含 `predicted_ignore_ratio`，只有 `evaluate_checkpoint.py` 事后统计；exp001 为 0.0，exp002 无记录。
- 建议：在 `evaluate()` 内按 epoch 统计 `predicted_ignore_ratio` 写入 history 作为质量门禁；mIoU 相近时优先选择该比例更低的 checkpoint；后续再按 docs 思路评估"8 通道重映射（0→255）"。

### P2 级（低风险 / 工程整洁）

**P2-1：存在三份并行的 mIoU 计算实现，易再次漂移**
- `train.py` `compute_miou()`（L101–112）在仓库中**未被任何代码调用**（死代码）；`train.py` `evaluate()`（L115–132）与 `evaluate_checkpoint.py` 又各有一份累计实现。
- 历史教训：旧版 `compute_miou`/`evaluate` 的"Ignore 未排除"bug 正源于重复实现未同步。建议收敛到单一可复用 evaluator（如 `metrics.py`），并加一个基于小样例的单元测试，保证三处调用同一实现。

**P2-2：`smoke_test.py` 的 loss 口径与训练不一致**
- L71 `cross_entropy(logits, mask_tensor)` 未带 `ignore_index=0`，也未做 ImageNet 归一化（对随机 Conv 冒烟无碍，但易误导）。建议与训练口径对齐，或在注释中说明差异是有意为之。

**P2-3：`evaluate_checkpoint.py` 通过 `from train import ...` 复用符号**
- L11 依赖运行时 sys.path 含 `src/`（需 `cd src` 或以 `python src/evaluate_checkpoint.py` 之外的方式运行）。建议后续将公共代码包化（`src/aic/`），消除隐式路径依赖。

**P2-4：`make_split.py` CSV 同时写入绝对路径列**
- L73–75 `image`/`mask` 列记录本机绝对路径；训练只用 `id` 列，功能无碍，但该列内容跨平台是"过期信息"，易让队友误读。建议只输出 `id`（或再附 data_root 的相对路径）。

**P2-5：文档存在近似重复且内容已漂移**
- `docs/suggestions/exp001_team_suggestions.md`（仓库根 suggestions/ 下）与 `docs/suggestions/exp001/exp001_team_suggestions.md` 是两份同一主题但已不同的副本（后者补充了 exp002 完成结果）。此外 `docs/suggestions/exp001/next_experiments.md` 建议编号跳号（3 → 5，缺 4）。
- 建议：保留 `docs/suggestions/expXXX/` 一份为准，删除或归档陈旧副本，避免 AI/队员把旧版当现状。

**P2-6：可复现性与元信息**
- `requirements.txt` 仅 4 个包且无版本；torch 按机器 CUDA 单独装（有意设计），但仓库未记录各实验实际的 python/torch/CUDA/transformers 版本（`run_config.json` 也不含 torch 版本）。建议在 `run_config.json` 补 `torch/transformers/cuda/python` 版本字段。
- 仓库无 LICENSE 文件。
- 随机性：`random/np/torch` 已固定种子（L164–166），但 DataLoader 未传 `generator`，多 worker 下增强/顺序的可复现性未完全锁定；如需严格复现建议补 generator 并记录 `cudnn.benchmark` 设置。

## 7. 改进建议路线

按"先稳后快、一次一个变量"原则（与团队既定纪律一致）：

| 优先级 | 动作 | 预期收益 |
|---|---|---|
| 立即（代码可靠性） | ① 补 `--resume`/last checkpoint；② 训练内记录 `predicted_ignore_ratio`；③ 收敛评价实现并加单测 | 中断不白跑；mIoU 口径可信 |
| 上线前（提交） | 新增 `predict.py` + 提交自检器；1024 分辨率验证重评 | 打通正式评测闭环 |
| exp003（单变量） | 加 LR 调度（warmup + cosine/poly），基准=exp002 768 | 末轮仍在涨，最可能低成本收益 |
| 后续候选 | 轻度颜色增强 → CE+Dice → 温和类别加权 → 8 通道重映射 → SegFormer B1/B2 | 分别针对泛化/不平衡/0 通道 |

注意保持同一 `train.csv`/`val.csv`，同一评价代码，以总 mIoU + 每类 IoU 共同决策（README 已明确此规则）。

## 8. 结论与限制

- **代码组织清晰、实验纪律好**：数据体检 → 固定划分 → 审计 → 训练 → 独立重评的分层合理；`id + data-root` 解耦路径、逐 epoch flush、best/summary 产物齐全，README 对代码/事实/建议分层的要求也有助于减少 AI 幻觉污染。
- **训练主流程健康**：SegFormer 基线在固定划分下 512→768 分辨率稳定提升 +0.01320（8 类全升），链路（CE ignore=0、同步翻转、AMP、修复后评价）正确。
- **主要风险集中在工程闭环**：无断点续训、无推理/提交脚本、训练中缺少对"有效像素预测为 0"比例的观测、评价实现重复。其中"预测为 0"的影响机制在原稿中表述有误，勘误后降级为监控项（见 P1-8 与附录 A）；其余三者不阻塞继续调模型，但应尽快补齐，否则前期实验成果在中断或提交环节可能折损。
- **限制**：本报告为静态代码 + 仓库内结果 JSON 的分析，未包含数据/权重，未实跑；数据侧细节（Ignore 占比、类别像素分布、近重复情况）依据 README 与审计结论转述，未独立复核；竞赛规则部分已依据官网公告逐条核对（来源与结论见附录 A）。

## 9. 附录 A：官网规则核对与勘误（2026-09-05）

### A.1 核对来源

| # | 来源 | URL |
|---|---|---|
| 1 | 赛题公告《2026AIC·无人机低空航拍图像语义分割》（AIC 大赛组委会，2026-04-30） | https://www.aicomp.cn/tracks/tracks-1/3708.html |
| 2 | 赛道通知《关于举办第八届全球校园人工智能算法精英大赛"算法挑战赛道"竞赛的通知》（全智赛组委会〔2026〕10 号，2026-04-29） | https://www.aicomp.cn/notice/notice-1/3629.html |
| 3 | 用户提供的报名/详情入口 | https://reg.aicomp.cn/app/JSGLPT/639980063d903c241eb84d5f |

说明：

- 来源 3 为 JS 渲染页面，直接抓取仅返回"当前浏览器不支持"，无法提取正文；本附录核对以上面两篇官网静态公告为准。
- 赛道导航页 https://www.aicomp.cn/tracks/tracks-1 亦列出"无人机低空航拍图像语义分割"位于**算法挑战赛道**（与来源 2 互相印证）。
- 官方赛题正文未给出 mIoU 公式图片、也未给出"类别 ID ↔ 名称"的数值映射表；这两项需以数据包（Label.txt / 样例 mask）为准。

### A.2 仓库假设 ↔ 官方规则逐条核对

1. **【一致】赛题归属与命名**：仓库 README 的"AIC 2026 无人机低空航拍图像语义分割"与官网赛题名一致；属《第八届全球校园人工智能算法精英大赛》（AIC）算法挑战赛道。官方同时明确该赛道采用"赛马制"：全国统一组织、不分赛区，复赛等同省级赛事，赛程为初赛 → 复赛 → 半决赛 → 总决赛。仓库未记录"赛马制/全国统一"等赛制背景——对代码无影响，但对排期与成绩判定有影响。

2. **【一致】训练数据规模**：官网"训练集共 6996 张图像"，且"所有图像已统一切分为 1024*1024 像素的 patch，并进行了重新排序"；仓库 README 记录正式训练集 6996 张图 + 6996 张 mask、全部 1024×1024 体检通过。两边一致。

3. **【补充】数据来源与测试发布节奏**：官网说明数据来自公开无人机数据集（含部分遥感图像）与南航工信部重点实验室自采数据，已统一清洗/类别映射；测试集分三次发布——测试集 1 含 500 张（初赛）、测试集 2 含 1300 张（复赛）、测试集 3 含 1794 张（与 1、2 合并用于半决赛），且各测试子集存在类别占比/跨域分布的显著差异。仓库无任何测试集规模信息；补齐 `predict.py`/提交脚本时应按官方各期目录与文件名组织（初赛 500 张），并严格满足"预测文件名与测试集文件名完全一致"。

4. **【一致，名称集合层面】类别体系**：官网"共包含 9 个语义类别（含背景及忽略类）：农田、水体、荒地、车辆、忽略、背景、建筑、道路、森林"。仓库映射为 0=Ignore、1=Background、2=Building、3=Road、4=Water、5=Barren、6=Vegetation、7=Agricultural、8=Vehicle，8 个有效类别参与 mIoU、忽略不参与。类别集合与官网一一对应（农田=Agricultural、水体=Water、荒地=Barren、车辆=Vehicle、忽略=Ignore、背景=Background、建筑=Building、道路=Road、森林=Vegetation；其中中文名 ↔ 仓库英文名的对应为语义推断，数值 ID 以数据包 Label.txt 为准）。
   **⚠ 注意**：官网列表中的名称顺序不代表像素 ID（其行文顺序是 农田/水体/荒地/车辆/忽略/背景/建筑/道路/森林）；若按"列表顺序当作 ID 顺序"理解，0=农田、4=忽略，将与仓库的 0=Ignore 冲突。仓库的 0=Ignore 依据应来自数据包 Label.txt（check_dataset.py 正是读 Label.txt）。提交前务必用官方样例 mask 复核一次"ID ↔ 中文名 ↔ 色块"，并在 predict.py 中固化该映射。

5. **【一致】评价指标**：官网"性能得分以 mIoU（平均交并比）作为评价指标……忽略类不参与计算"。仓库在 8 个有效类别（1~8）上算 mIoU、忽略 GT 像素，与规则文字一致（官网未附公式图，无法做逐像素级进一步对照）。

6. **【一致，且要求更严】提交格式**：官网要求预测 PNG 与输入图像**文件名完全一致**；像素值**严格对应类别 ID（0-8）**；**单通道 PNG**，严禁 RGB 彩色图与带调色板（Palette）的索引色 PNG；分辨率必须与原测试图（均 1024×1024）**严格一致**，评测系统不做任何缩放/裁剪，凡尺寸或格式不符直接判无效/错误预测并记 0 分；最终压缩为 zip 提交。仓库 README 已写"同名、1024×1024、单通道 PNG、像素值为合法类别 ID"，方向一致；但"不符即 0 分""zip 打包""自动自检器"尚未落地——进一步印证正文 P0-2/P1-4：predict.py + 提交校验器 + 1024 口径正式重评应作为上线前必做项。

7. **【一致，且为硬约束】算法与数据约束**：预训练只能使用学术领域公开权重的模型做微调，不能使用商业闭源模型/API；不能使用额外数据；**不允许多模型集成（model ensemble），最终提交只含一个模型**；半决赛结束后赛事方将用官方数据复现提交代码，无法复现将取消成绩。仓库当前为单个 SegFormer（HF 公开学术权重）微调、无额外数据、无集成，符合约束。后续路线图应继续避免任何集成式手段（如多模型平均/bagging，若被认定 ensemble 存在违规风险）。

8. **【补充】成绩判定与官方基线**：初赛（测试集 1）机器自动评分 mIoU、**不计入决赛总分**，mIoU>0 即可进入复赛；复赛（测试集 2）同样不计入总分；半决赛（测试集 3 + 1 + 2 合并）计入决赛总分；且"若参赛者提交结果低于赛事方公布的基线成绩，赛事方有权将其认定为无效成绩"。本次抓取的两篇公告未公布官方基线数值，建议关注赛事群/官网后续公告；仓库应在本地维护一条"1024 口径验证集 mIoU"与官方基线对照，避免辛苦提升后仍低于官方基线被判无效。

9. **【一致】技术栈与参考资源**：官网要求用 Python 与开源框架（如 PyTorch），并在参考资源中直接列出 SegFormer（NeurIPS 2021）、MMSegmentation、SAM3；仓库选 Hugging Face SegFormer MIT-B0 微调，与官方建议一致。

10. **【排期参考】时间线**：报名开启 2026-04-28；2026 年 9 月中旬前开始复赛；2026-10-10 前开始半决赛；总决赛计划 11 月中下旬。报告日期 2026-09-05 正处于初赛评分/复赛启动窗口，建议优先用 exp002 best_model 打通初赛测试集（500 张）推理与提交，并同步准备"完整训练 + 验证代码 + 环境文档"的 docker 复现材料。

### A.3 对正文的勘误与修订记录

- **修订 1（原 P0-3 → 现 P1-8）**：原稿 P0-3 称"有效像素被判为 0 时既不进 TP 也不进 FP、这类错误被'隐形'处理、mIoU 会被系统性高估"。逐行核对 `train.py`（`evaluate()` L124–129、`compute_miou()` L104–110）后确认：GT=c 而预测=0 的像素经 `actual = (targets == c) & valid` 仍计入类别 c 的 union、但不进 intersection，等价于类别 c 的一个 FN，会照常降低 IoU_c——预测成 0 并不"隐形"，mIoU 也不会因此被系统性高估。该项已**降级为监控项 P1-8 并迁入 P1 节**，P0 节只留勘误注。
- **修订 2（第 1 章）**：补充赛道归属与"赛马制"四阶段赛制背景。
- **修订 3（第 8 章）**：风险句同步改为"缺少对有效像素预测为 0 比例的观测"，并注明机制已勘误。
- **修订 4（第 8 章限制段）**：补注"竞赛规则已依据官网公告核对，见附录 A"。

### A.4 仍未决事项（需数据包/赛事方进一步确认）

1. 类别数值 ID ↔ 名称映射：以官方数据包 Label.txt / 样例 mask 复核为准，勿用官网名称列表顺序推断。
2. 官方基线 mIoU 数值（本次抓取的两篇公告未公开）。
3. 官方评测代码级口径（"预测=0 落在有效像素"按标准 FN 处理是与常见实现一致的假设，尚无官方代码佐证）。
4. 数据包中 Ignore 像素占比与空间分布（决定"0→255 重映射"实验是否值得做）。
5. 半决赛 docker 复现模板与提交要求细节（以组委会后续正式通知为准）。

---

*报告生成日期：2026-09-05；附录 A（官网核对与勘误）更新于 2026-09-05。*



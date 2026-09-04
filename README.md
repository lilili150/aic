# AIC 2026 无人机低空图像语义分割

本项目参加 AIC 2026 无人机低空航拍图像语义分割赛道。模型输入 RGB 航拍图，输出同尺寸的单通道类别 ID 图。

## 当前进度

- 正式训练集已体检通过：6996 张图片和 6996 张 mask，全部为 `1024 x 1024`。
- 已生成固定划分：训练集 5597 张，验证集 1399 张，随机种子为 `2026`。
- SegFormer MIT-B0 基线已在 AutoDL RTX 5090 上完成 30 epoch 训练。
- 训练脚本已支持逐 epoch 持久化：即使训练中断，也能从 `history.jsonl` 和 `history.json` 恢复已完成记录。
- 首轮最佳验证集 mIoU 为 `0.74514`（epoch 25），完整分析见 `docs/analysis/exp001_baseline.md`。

## 类别和评价

| ID | 类别 |
|---:|---|
| 0 | Ignore |
| 1 | Background |
| 2 | Building |
| 3 | Road |
| 4 | Water |
| 5 | Barren |
| 6 | Vegetation |
| 7 | Agricultural |
| 8 | Vehicle |

类别 0 `Ignore` 不参与 mIoU，也不参与训练 loss；有效评价类别为 1～8。预测结果必须是同名、`1024 x 1024`、单通道 PNG，像素值为合法类别 ID。

## 目录说明

```text
src/                 可运行代码
  check_dataset.py   检查图片、mask、尺寸、配对和标签值
  make_split.py      生成固定 train/validation 划分
  smoke_test.py      检查环境、CUDA、Tensor 和基础训练链路
  train.py           SegFormer 训练、验证和实验记录
outputs/results/     各实验的原始结果，例如 history.json
outputs/splits/      固定划分文件（本地或云端生成）
docs/analysis/       基于真实结果的分析记录
docs/suggestions/    下一步实验建议和决策依据
```

队友使用 AI 时，先让 AI 阅读 `README.md`、`src/`、`outputs/results/`、`docs/analysis/` 和 `docs/suggestions/`，再提出分析或改代码。这样代码、事实结果和主观建议彼此分开，避免 AI 把建议误当成实验事实。

正式数据、示例数据、预训练权重、虚拟环境和训练运行目录不上传公开 GitHub，具体规则见 `.gitignore`。队友应通过私有云盘、私有服务器或 AutoDL 持久化目录获得数据访问权限。

## 基线配置

| 项目 | 值 |
|---|---|
| 模型 | Hugging Face SegFormer MIT-B0 |
| 输出类别 | 9 |
| 输入尺寸 | `512 x 512` |
| batch size | 2 |
| 优化器 | AdamW |
| 学习率 | `6e-5` |
| weight decay | `0.01` |
| loss | CrossEntropy，忽略类别 0 |
| 增强 | 同步水平翻转 |
| 归一化 | ImageNet mean/std |
| AMP | CUDA 上开启 |

## 本地数据检查

```powershell
python src/check_dataset.py `
  --images "2026-低空图像语义分割赛道-训练集/train/train/images" `
  --masks "2026-低空图像语义分割赛道-训练集/train/train/masks" `
  --labels "2026-低空图像语义分割赛道-训练集/Label.txt" `
  --report outputs/official_train_dataset_report.json
```

生成固定划分：

```powershell
python src/make_split.py `
  --images "2026-低空图像语义分割赛道-训练集/train/train/images" `
  --masks "2026-低空图像语义分割赛道-训练集/train/train/masks" `
  --labels "2026-低空图像语义分割赛道-训练集/Label.txt" `
  --output-dir outputs/splits `
  --val-ratio 0.2 `
  --seed 2026
```

## AutoDL 训练

训练前准备好云端目录 `/root/autodl-tmp/dataset`、`/root/autodl-tmp/train.csv`、`/root/autodl-tmp/val.csv` 和本地模型 `/root/autodl-tmp/models/mit-b0`。离线训练命令如下：

```bash
export HF_HUB_OFFLINE=1
python src/train.py \
  --data-root /root/autodl-tmp/dataset/train/train \
  --train-csv /root/autodl-tmp/train.csv \
  --val-csv /root/autodl-tmp/val.csv \
  --pretrained /root/autodl-tmp/models/mit-b0 \
  --output-dir /root/autodl-tmp/runs/exp002_baseline_logged \
  --epochs 30 \
  --device cuda \
  --log-file /root/autodl-tmp/runs/exp002_baseline_logged/training.log
```

每个实验目录会产生 `run_config.json`、`history.jsonl`、`history.json`、`best_metrics.json`、`best_metrics.pt`、`best_model/`、`summary.json` 和 `training.log`。其中 JSONL 每个 epoch 一行并立即 flush，适合实时追踪；JSON 是可直接读取的完整快照。

```bash
tail -n 5 /root/autodl-tmp/runs/exp002_baseline_logged/history.jsonl
cat /root/autodl-tmp/runs/exp002_baseline_logged/summary.json
```

不要覆盖已完成的实验目录。每个改动使用新的实验编号，并保持同一份 `train.csv` 和 `val.csv`，这样 mIoU 才能公平比较。

## 首轮结果分析

首轮 30 epoch 的最佳验证集 mIoU 为 `0.74514`（epoch 25），第 30 轮为 `0.74312`。第 25～30 轮已经进入平台期，因此后续推理应使用最佳 checkpoint，而不是盲目使用最后一轮。`ious` 数组索引始终对应类别 ID，索引 0 为 `null`，不能把它纳入平均值。

首轮最低的有效类别是 `Barren`（最佳 IoU `0.53340`），其次是 `Background`（`0.66254`）和 `Vehicle`（`0.71723`）。完整的每类数据和判断记录在 `docs/analysis/exp001_baseline.md`。

如果需要从 AutoDL 重新下载原始记录：

在 Windows PowerShell 中，可用 AutoDL 提供的 SSH 端口把首轮记录下载到本项目：

```powershell
New-Item -ItemType Directory -Force outputs/results/exp001_baseline | Out-Null
scp -P 42544 root@connect.weste.seetacloud.com:/root/autodl-tmp/runs/exp001_baseline_ignore0/history.json outputs/results/exp001_baseline/history.json
scp -P 42544 root@connect.weste.seetacloud.com:/root/autodl-tmp/runs/exp001_baseline_ignore0/summary.json outputs/results/exp001_baseline/summary.json
```

若云端实验目录名称不同，先执行 `find /root/autodl-tmp/runs -maxdepth 3 -type f | sort`，再替换命令中的目录名。

## GitHub 上传边界

建议上传：`README.md`、`.gitignore`、`requirements.txt`、`src/check_dataset.py`、`src/make_split.py`、`src/smoke_test.py`、`src/train.py`。

禁止上传正式比赛数据和示例数据、`.venv/`、`.venv-1/`、缓存、`models/` 下的预训练权重、`runs/`、checkpoint、日志、预测结果，以及 SSH 私钥、密码、Token 和 `.env` 文件。完整训练 CSV 和比赛数据报告也建议留在本地。

`requirements.txt` 只记录 Python 层依赖；PyTorch 应根据云端 CUDA 版本单独安装。不要把带有特定机器 CUDA wheel 的环境目录提交到仓库。

## 后续实验原则

先完整保存并分析 `exp001`，再一次只改变一个因素，例如轻度颜色增强、Dice Loss、类别加权或输入尺寸。所有实验必须使用相同固定划分、相同评价代码，并以验证集 mIoU 和每类 IoU 共同决定是否保留。

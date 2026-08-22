# 服务器测试命令

本文档只列当前仓库能够执行的命令。运行前先按 [Linux 部署指南](testing_linux.md) 安装 CANN、`torch_npu`、MindIE-SD、DiffSynth 和 Hunyuan 依赖，并确认模型权重和 COYO metadata 已在服务器本地。

## 0. 公共环境

```bash
cd /path/to/HunyuanImage3-SLA
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:$PWD/train:$PWD/upstream/DiffSynth-Studio:$PWD/upstream/MindIE-SD:$PWD/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
npu-smi info
```

## 1. 环境和 kernel

```bash
python - <<'PY'
import torch
import torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
PY

python -m pytest -q upstream/MindIE-SD/tests/layers/flash_attn/test_sparse_linear_attn.py
```

## 2. COYO 候选、下载和多 NPU VAE-only 离线采样

修改 `configs/sampling.yaml` 的权重路径、原始 manifest 和图片目录，然后执行。`12,000` 是候选数，不是最终训练数量；采样器会得到首批成功的 `2,000` 条。

```bash
python tools/select_coyo_subset.py \
  --input /datasets/coyo/metadata.jsonl \
  --output /datasets/hunyuan_sla/candidates.jsonl \
  --candidate-count 12000

python tools/download_images.py \
  --metadata /datasets/hunyuan_sla/candidates.jsonl \
  --output-dir /datasets/hunyuan_sla/raw

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 bash scripts/sample.sh configs/sampling.yaml --resume
python tools/inspect_latent.py --cache-dir data/cache
```

采样器只加载 tokenizer、图像预处理和 VAE，不创建 Hunyuan Transformer/MoE。它按 `int(sample_id) % world_size` 静态分片；非数字 ID 使用 SHA-256 的稳定分片。每个 rank 写入 `data/cache/rank-XXX/`，rank 0 用硬链接合并为 `data/cache/shards/`、生成总 `manifest.jsonl` 并写入 `READY.json`。每个 rank 目标数是 `target_count` 的均分；必须准备足够候选，保证每个分片都有可用图片。

## 3. 单 NPU Dense 与 SLA one-step

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_dense.sh configs/train_sla.yaml --max-steps 1
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

两个命令都必须出现 `loss=... finite_grad=True`，并在 `results/training/default/` 生成 checkpoint。日志分别写入 `logs/training/`。

## 4. 多 NPU DDP one-step

先完成单卡 SLA one-step。当前训练入口具备 Accelerate/DDP 启动路径：每个 rank 保存完整模型副本、读取不同 cache 样本，只有 rank 0 写 checkpoint。它不是 80B 模型的内存解决方案，只有单张 NPU 能容纳完整 BF16 checkpoint、Dense teacher 前向和 SLA student 反向时才可用；当前仓库尚未在多 NPU 实机验证。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

不要将 `NPROC_PER_NODE` 大于可见 NPU 数量。当前不支持 sequence parallel（SP）、tensor parallel（TP）、expert parallel（EP）或 ZeRO；不得把对应的 DeepSpeed/xDiT 参数加入此命令。若单卡放不下模型，不要直接执行本节命令，应先完成并行训练适配。

## 5. 断点恢复

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --max-steps 200 \
  --resume-from results/training/default/sla-step-100.pt
```

恢复时必须使用相同的 NPU 数、相同 cache、相同配置和相同 batch 语义。

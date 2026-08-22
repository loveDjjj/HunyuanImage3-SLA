# 服务器测试命令

本文档列出当前仓库的服务器命令。以下第 0 至第 2 节只用于 VAE-only 离线采样，不安装 DiffSynth-Studio、MindIE-SD 或 SLA kernel。CANN 8.5 已由服务器提供，模型权重路径固定为 `/mnt/weight/HunyuanImage-3.0-Instruct-Distil`。

## 0. 创建 VAE-only Conda 环境

```bash
cd /path/to/HunyuanImage3-SLA

# 不修改已有 cann-8.5、aisbench_npu 环境；按绝对路径新建采样环境。
conda create -y -p /mnt/share/r50063443/conda_envs/hunyuan-vae python=3.10
conda activate /mnt/share/r50063443/conda_envs/hunyuan-vae
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# CANN 8.5.0 + x86_64：CPU 版 PyTorch 加 Ascend torch_npu 插件。
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch==2.9.0+cpu' --index-url https://download.pytorch.org/whl/cpu

# 将文件名替换为已下载、且与 Python 3.10 / x86_64 匹配的官方 wheel 实际路径。
python -m pip install /mnt/wheels/torch_npu-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl

# VAE-only 需要的 Hunyuan Python 依赖与项目工具依赖。
python -m pip install -r upstream/HunyuanImage-3.0/requirements.txt
python -m pip install safetensors pillow pyyaml tqdm requests pyarrow
python -m pip install -e upstream/HunyuanImage-3.0

export PYTHONPATH="$PWD:$PWD/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
```

如果服务器是 `aarch64`，将 PyTorch 安装命令替换为 `python -m pip install torch==2.9.0`，并使用名称含 `aarch64` 的 `torch_npu` 2.9.0 wheel。Python 3.11 必须使用 `cp311` wheel，不能使用上例的 `cp310`。

## 1. 验证 VAE-only 环境

```bash
npu-smi info
python - <<'PY'
import torch
import torch_npu
assert torch.npu.is_available()
print(torch.__version__, torch_npu.__version__, torch.npu.device_count())
PY
```

## 2. COYO 候选、下载和多 NPU VAE-only 离线采样

`configs/sampling.yaml` 已预设权重路径 `/mnt/weight/HunyuanImage-3.0-Instruct-Distil`；只需修改原始 manifest 和图片目录。`12,000` 是候选数，不是最终训练数量；采样器会得到首批成功的 `2,000` 条。

先下载 COYO metadata。COYO 不包含预下载图片，而是发布图片 URL、caption 和质量元数据。无需下载完整 747M 数据集；下面只下载官方的一个 Parquet shard。该 shard 含数百万行，足以选出 12K 个候选，但由于 COYO URL 较旧，图片下载阶段仍会丢失部分样本。

```bash
mkdir -p /datasets/coyo
wget -c \
  'https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00000-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet' \
  -O /datasets/coyo/coyo-part-00000.parquet
```

```bash
python tools/select_coyo_subset.py \
  --input /datasets/coyo/coyo-part-00000.parquet \
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

以下命令**不能**在 `hunyuan-vae` 环境执行。它们需要单独的训练 Conda 环境，其中包含 DiffSynth-Studio、MindIE-SD、Triton-Ascend、Accelerate 和训练所需依赖；完整安装步骤见 [testing_linux.md](testing_linux.md)。

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

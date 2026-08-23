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
# torch 2.9.0 必须配套 torchvision 0.24.0；两者都使用 CPU wheel，NPU 后端由 torch_npu 提供。
python -m pip install 'torchvision==0.24.0' --index-url https://download.pytorch.org/whl/cpu

# 将文件名替换为已下载、且与 Python 3.10 / x86_64 匹配的官方 wheel 实际路径。
python -m pip install /mnt/wheels/torch_npu-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl

# VAE-only 需要的 Hunyuan Python 依赖与项目工具依赖。
python -m pip install -r upstream/HunyuanImage-3.0/requirements.txt
python -m pip install safetensors pillow pyyaml tqdm requests huggingface_hub
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

python - <<'PY'
import torch
import torchvision
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
assert torch.__version__.startswith('2.9.0')
assert torchvision.__version__.startswith('0.24.0')
PY
```

若 `import torchvision` 报 `operator torchvision::nms does not exist`，说明环境中混入了不匹配的 torchvision（常见为 0.20.x 或 CUDA wheel）。在已激活 `hunyuan-vae` 环境中修复，不要重装 `torch_npu`：

```bash
python -m pip uninstall -y torchvision
python -m pip install --no-deps --no-cache-dir --force-reinstall \
  'torchvision==0.24.0' \
  --index-url https://download.pytorch.org/whl/cpu
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'
```

## 2. Flickr30k 准备和多 NPU VAE-only 离线采样

`configs/sampling.yaml` 已预设权重路径及 Flickr30k 本地路径。数据放在仓库的 `datasets/flickr30k/`，该目录已由 Git 忽略。

当前已下载的 Flickr30k 文件应位于：

```text
/mnt/share/r50063443/HunyuanImage3-SLA/datasets/flickr30k/
  flickr30k-images.tar
  dataset_flickr30k.json
```

其中 `flickr30k-images.tar` 是图像 archive，`dataset_flickr30k.json` 提供图片文件名、train/val/test split 和每图五条英文 caption。`gitattributes` 与当前流程无关；`flickr30k.tar.gz` 不需要使用。Flickr30k 原图受 Flickr 条款约束，只应按其研究/教育许可使用。

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
tar -tf datasets/flickr30k/flickr30k-images.tar | head
tar -xf datasets/flickr30k/flickr30k-images.tar -C datasets/flickr30k
test -d datasets/flickr30k/flickr30k-images
```

```bash
python tools/prepare_flickr30k_manifest.py \
  --annotations datasets/flickr30k/dataset_flickr30k.json \
  --images-dir datasets/flickr30k/flickr30k-images \
  --output datasets/flickr30k/metadata.jsonl \
  --split train \
  --sample-count 2000

wc -l datasets/flickr30k/metadata.jsonl
head -n 1 datasets/flickr30k/metadata.jsonl
```

manifest 工具固定随机种子，先保证每图只保留一条确定性 caption，再从训练 split 抽取 2,000 张唯一图片。不要将同一图的五条 caption 当作五个训练样本，否则 `latent_z0` 会重复五次。

执行 16 卡 latent 采样：

```bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NPROC_PER_NODE=16 bash scripts/sample.sh --resume
python tools/inspect_latent.py --cache-dir data/cache
```

采样器只加载 tokenizer、图像预处理和 VAE，不创建 Hunyuan Transformer/MoE。它按 `int(sample_id) % world_size` 静态分片；每个 rank 写入 `data/cache/rank-XXX/`，rank 0 用硬链接合并为 `data/cache/shards/`、生成总 `manifest.jsonl` 并写入 `READY.json`。

## 3. 单 NPU Dense 与 SLA one-step

以下命令**不能**在 `hunyuan-vae` 环境执行。它们需要单独的训练 Conda 环境，其中包含 DiffSynth-Studio、MindIE-SD、Triton-Ascend、Accelerate 和训练所需依赖；完整安装步骤见 [testing_linux.md](testing_linux.md)。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_dense.sh configs/train_sla.yaml --max-steps 1
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

两个命令都必须出现 `loss=... finite_grad=True`，并在 `results/training/default/` 生成 checkpoint。日志分别写入 `logs/training/`。

## 4. 16 NPU ZeRO-3 one-step

训练环境需要额外安装 DeepSpeed。以下命令使用 Accelerate + DeepSpeed ZeRO-3，在 16 卡间切分 80B 模型参数、梯度与 optimizer state：

```bash
python -m pip install deepspeed
python -c 'import accelerate, deepspeed; print(accelerate.__version__, deepspeed.__version__)'

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

预期日志包含 `step=1 loss=... finite_grad=True`，并生成目录 `results/training/default/sla-step-1/`。这条路径已经完成代码接入，但尚未在 910C A3 实机验收。若加载阶段 OOM，首先确认日志显示 `DistributedType.DEEPSPEED`，而不是 DDP；如果 Hunyuan 自定义 MoE 与 ZeRO hook 不兼容，需要根据首个完整 traceback 继续适配。

## 5. 断点恢复

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --max-steps 200 \
  --resume-from results/training/default/sla-step-100
```

恢复时必须使用相同的 NPU 数、相同 cache、相同配置和相同 batch 语义。

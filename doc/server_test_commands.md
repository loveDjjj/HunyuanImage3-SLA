# 服务器测试命令

本文档列出当前仓库的服务器命令。以下第 0 至第 2 节只用于 VAE-only 离线采样，不安装 DiffSynth-Studio、MindIE-SD 或 SLA kernel。CANN 8.5 已由服务器提供，模型权重路径固定为 `/mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil`。

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
# 先只验证所有 rank 都能构造 VAE 并加载权重，不处理图片。
NPROC_PER_NODE=16 bash scripts/sample.sh --load-only

# load-only 输出 vae_load_ok=True 后再正式采样。
NPROC_PER_NODE=16 bash scripts/sample.sh --resume
python tools/inspect_latent.py --cache-dir data/cache
```

采样器只加载 tokenizer、图像预处理和 VAE，不创建 Hunyuan Transformer/MoE。它按 `int(sample_id) % world_size` 静态分片；每个 rank 写入 `data/cache/rank-XXX/`，rank 0 用硬链接合并为 `data/cache/shards/`、生成总 `manifest.jsonl` 并写入 `READY.json`。

ModelScope 发布的 Instruct-Distil `config.json` 可能没有 `model_version`。采样器会使用当前检出的 HunyuanImage-3.0 配置类解析该文件，并为 Distil checkpoint 使用 `HunyuanImage-3.0-Instruct` tokenizer 布局；不要手工修改 168GB 权重目录。若仍出现 `model_version` 错误，先执行 `git pull` 并确认 `sampling/vae_only.py` 中不再使用 `AutoConfig`。

上游 `AutoencoderKLConv3D` 构造函数包含一个仅供 decode 使用的空 CUDA sentinel。VAE-only adapter 在构造期间将它重定向到当前 NPU；encode 路径不调用上游的 CUDA-only `decode_dist()`。日志中的 `PreTrainedTokenizerFast` 与 `HunyuanImage3TokenizerFast` 类型提示来自官方 tokenizer metadata，当前加载结果仍是 Hunyuan 子类，不是本次 CUDA 异常的原因。

## 3. 单 NPU Dense 与 SLA one-step

以下命令**不能**在 `hunyuan-vae` 环境执行。它们需要单独的训练 Conda 环境，其中包含 DiffSynth-Studio、MindIE-SD、Triton-Ascend、Accelerate 和训练所需依赖；完整安装步骤见 [testing_linux.md](testing_linux.md)。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_dense.sh configs/train_sla.yaml --max-steps 1
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

两个命令都必须出现 `loss=... gradient_elements=... gradient_norm=... finite_grad=True`。
当前主配置还必须分别出现 `proj_l_grad_norm`、`qkv_delta_grad_norm` 和
`o_delta_grad_norm`，checkpoint 写入 `results/training/qkvo-delta/`。

## 4. 16 NPU ZeRO-3 one-step

训练环境需要额外安装 DeepSpeed。以下命令使用 Accelerate + DeepSpeed ZeRO-3，在 16 卡间切分 80B 模型参数、梯度与 optimizer state：

```bash
python -m pip install deepspeed
python -c 'import accelerate, deepspeed; print(accelerate.__version__, deepspeed.__version__)'

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

QKV/O 主配置必须保持 `BLKQ=128, BLKK=128`。旧的 `BLKQ=64, BLKK=128` 在
AscendC proj-only 路径可运行，但强制 Triton 后会在 910C 编译阶段 UB overflow。

预期生成目录 `results/training/qkvo-delta/sla-step-1/`。此前记录的
`gradient_elements=528384` 是 proj-only baseline 的实机验收，不代表新的 QKV/O delta
已经验收。新主配置理论总训练参数为 `1,342,705,664`，必须确认三个参数组均产生
非零有限梯度后才能开始长训。

ZeRO-3 在 backward 后会归约和切分梯度，并可能清空普通的 `parameter.grad`。项目因此在 `backward()` 和 `optimizer.step()` 之间使用 DeepSpeed 的 `safe_get_local_grad()` 检查本 rank 梯度分片，然后跨 rank 汇总梯度元素数、非有限值数量和 L2 norm。不要用 `parameter.grad is not None` 判断 ZeRO-3 是否产生梯度，这会把正常的分片梯度误报为缺失。`gradient_elements` 必须大于 0，`gradient_norm` 必须是有限数。

训练启动时会打印实际日志路径，进度条只在主 rank 显示：

```text
training_log=/mnt/share/r50063443/HunyuanImage3-SLA/logs/training/<时间>.log
sla training: ...
```

查看最新日志和 checkpoint 分片：

```bash
ls -lht logs/training | head
tail -f "$(ls -1t logs/training/*.log | head -1)"
du -sh results/training/qkvo-delta/sla-step-1
du -h results/training/qkvo-delta/sla-step-1/* | sort -h
cat results/training/qkvo-delta/latest
```

16 卡 ZeRO-3 每个 checkpoint 有 16 个 `model_states.pt` 和 16 个 `optim_states.pt`。前者保存每个 rank 的模型 checkpoint 状态和训练参数分片信息，后者保存 FP32 master parameter、Adam moments 及优化器分片；`latest` 是最新 tag 指针，`zero_to_fp32.py` 是合并辅助脚本。断点恢复必须保留同一 step 的全部 rank 文件。

如果保存时报 `Parent directory .../sla-step-N does not exist`，这是旧版本没有在 collective save 前显式创建 tag 目录造成的多 rank 竞态。更新后每个 rank 都会幂等创建绝对 checkpoint 路径，并在写入前 barrier：

```bash
git pull origin main

cat results/training/default/latest
for dir in results/training/default/sla-step-*; do
  printf '%s model=%s optimizer=%s\n' "$dir" \
    "$(find "$dir" -maxdepth 1 -name '*model_states.pt' | wc -l)" \
    "$(find "$dir" -maxdepth 1 -name '*optim_states.pt' | wc -l)"
done
```

只有同时包含 16 个 model shard 和 16 个 optimizer shard 的目录才可恢复。删除不完整的失败目录，从最近的完整 checkpoint 继续。例如 `sla-step-90` 完整时：

```bash
rm -rf results/training/default/sla-step-100
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla --max-steps 200 \
  --resume-from results/training/default/sla-step-90
```

日志中 `mindiesd::block_sparse_attention` 的 Autograd 注册警告不会阻止当前 one-step，但正式长训前仍需做多步 loss 和参数更新量验证。当前有限梯度证明执行链路打通，不单独证明自定义算子的梯度数值精度。

如果 `--max-steps 200` 在 step 125 正常保存后退出，原因是旧循环把 `num_epochs: 1` 当成硬上限：2,000 条数据经过 16 rank 切分后，每个 rank 的一个 epoch 正好是 125 step。更新后 `max_steps` 是停止目标，代码会自动把有效 epoch 数扩展为 2，并打印：

```text
batches_per_epoch=125 configured_epochs=1 effective_epochs=2 max_steps=200
```

已有的 `sla-step-125` 若包含完整的 16+16 分片，可以直接继续：

```bash
git pull origin main
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla --max-steps 200 \
  --resume-from results/training/default/sla-step-125
```

离线 cache 已经包含 `latent_z0`，训练 forward 不执行 VAE encode，也没有条件图片需要 ViT。`configs/train_sla.yaml` 因此默认通过 Hunyuan 上游的 `skip_load_module` 跳过 `vae` 和 `vit`，避免加载冻结权重并绕过上游 VAE 构造函数中的 `device="cuda"` sentinel。若日志仍在 `autoencoder_kl_3d.py:502` 报 `Torch not compiled with CUDA enabled`，说明服务器代码尚未更新：

```bash
git pull origin main
grep -A3 '^skip_load_modules:' configs/train_sla.yaml

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

加载日志列出 `vae.*`、`vision_model.*` 或 `vision_aligner.*` 为未使用 checkpoint 权重属于预期现象，因为这些模块已被主动跳过；Transformer、diffusion input/output 层或 SLA 目标层不应出现在该列表中。

latent cache 的每条 record 已经是一个完整 micro batch。训练 DataLoader 使用 `batch_size=1` 向 Accelerate 提供标准 batch sampler，并通过自定义 collate 直接返回唯一 record，因此不会给缓存 Tensor 增加额外维度。若日志在 `len(dataloader)` 报 `batch_sampler` 为 `None`，先执行 `git pull origin main`。

DeepSpeed 同样需要知道每卡 micro batch 的语义大小。Accelerate launch YAML 会丢弃部分扩展 DeepSpeed 字段，因此 `configs/train_sla.yaml` 保存 `train_micro_batch_size_per_gpu: 1`，训练入口在 `accelerator.prepare()` 前将它直接写入 active DeepSpeed plugin。若 `_prepare_deepspeed` 报该字段缺失，更新代码并检查配置：

```bash
git pull origin main
grep '^train_micro_batch_size_per_gpu:' configs/train_sla.yaml
```

修复后的主 rank 日志必须在模型加载前打印 `deepspeed_train_micro_batch_size_per_gpu=1`。

Instruct-Distil checkpoint 同时启用 CFG distillation 和 MeanFlow。训练会根据缓存中的 `guidance_index` 动态传入官方 guidance scale `2.5` 对应的 embedding 值 `2500`，并为 `timesteps_r_index` 采样满足 `r <= t` 的 MeanFlow 第二 timestep；这两个数值不写入离线 cache。若 forward 在 `instantiate_guidance_tokens` 或 `instantiate_timestep_r_tokens` 收到 `None`，先更新代码并确认 `configs/train_sla.yaml` 包含 `conditioning.guidance_scale: 2.5`。

训练直接调用 diffusion forward，不经过 Hunyuan 的 `generate()`。入口会根据 `image_mask` 和静态 condition index 初始化 `post_token_len`、`num_image_tokens`、`num_special_tokens`，并设置 `use_cache=False`。若报模型缺少这些 runtime 属性，先执行 `git pull origin main`。

默认性能配置将 ZeRO-3 parameter/optimizer shard 常驻 NPU，并保留逐层 non-reentrant
activation checkpointing。训练日志应显示两个 offload device 均为 `none`，以及
`activation_checkpointed_layers=32`。`scripts/train.sh` 仍设置
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。

```bash
grep -E 'offload_(param|optimizer)_device' configs/accelerate_zero3_16npu.yaml
grep '^activation_checkpointing:' configs/train_sla.yaml
```

预期分别为 `none` 和 `true`。建议持续监控峰值 HBM并保留至少 8GiB 余量。若 OOM：

```bash
ACCELERATE_CONFIG=configs/accelerate_zero3_16npu_offload.yaml \
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla --max-steps 5 --output-dir results/training/qkvo-offload-fallback
```

Activation checkpoint 的 backward 会在原始 forward 返回后重算 decoder layer。checkpoint wrapper 会在初次计算和重算阶段分别建立 CUDA→NPU runtime 兼容上下文；若 backward 重算仍在 `torch.cuda.set_device` 报 `_cuda_setDevice` 不存在，先执行 `git pull origin main`。

## 5. 断点恢复

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --max-steps 200 \
  --resume-from results/training/qkvo-delta/sla-step-100
```

恢复时必须使用相同的 NPU 数、相同 cache、相同配置和相同 batch 语义。

## 6. 导出 vLLM-Omni SLA 权重

训练完成并确认 `latest` 指向完整 checkpoint 后，在同一个训练环境中执行：

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
git pull origin main

cat results/training/qkvo-delta/latest
bash scripts/export_sla_adapter.sh
```

脚本默认读取 `results/training/qkvo-delta/latest`，并输出：

```text
results/adapters/<latest-tag>/adapter.safetensors
results/adapters/<latest-tag>/adapter_config.json
results/adapters/<latest-tag>/SHA256SUMS
```

正式交付建议显式指定最终 step。例如训练到 200 step：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/qkvo-delta/sla-step-200 \
  results/adapters/qkvo-delta-step-200
```

再次导出同一路径时增加 `--force`：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/qkvo-delta/sla-step-200 \
  results/adapters/qkvo-delta-step-200 \
  --force
```

单独验证：

```bash
python tools/inspect_sla_adapter.py \
  --adapter-dir results/adapters/qkvo-delta-step-200

cd results/adapters/qkvo-delta-step-200
sha256sum -c SHA256SUMS
```

成功结果必须包含：

```text
"valid": true
"tensor_count": 128
"parameter_count": 1342705664
"dtype": ["torch.bfloat16", "torch.float32"]
```

提供给 vLLM-Omni 的 adapter 路径：

```text
/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/qkvo-delta-step-200/adapter.safetensors
```

基础模型仍由 vLLM-Omni 从
`/mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil` 单独加载。导出成功并通过
SHA256 检查前，不要删除 ZeRO-3 checkpoint 中的 `optim_states.pt`。

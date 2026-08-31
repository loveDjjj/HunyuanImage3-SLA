# HunyuanImage3-SLA

HunyuanImage3-SLA 是一个面向 Ascend NPU 的 HunyuanImage-3.0-Instruct-Distil SLA 恢复训练项目。

项目以 DiffSynth-Studio 的 `DiffusionTrainingModule` 为训练模块基础，在不重写 HunyuanImage3 模型、也不重新实现 MindIE-SD kernel 的前提下，将 Hunyuan diffusion Transformer 中的 Dense GQA attention 运行时替换为 MindIE-SD `SparseLinearAttention`。

## 目标

- 复用 HunyuanImage-3.0-Instruct-Distil。
- 在 Ascend NPU 上执行 SLA forward、backward 和 fine-tuning。
- 保持 Hunyuan GQA 结构：32 个 Q heads、8 个 KV heads。
- 复用 Hunyuan 原有 `repeat_kv`，将 K/V 扩展为 32 heads 后输入 SLA。
- 使用 Dense Hunyuan `diffusion_prediction` 作为 teacher，SLA Hunyuan 输出作为 student，以 MSE 做 recovery training。
- 保留 `proj_l`-only baseline，并支持固定 Dense teacher 下的全秩 QKV/O delta adaptation。

## 目录

```text
train/
  train_sla.py        # Dense/SLA 单步与多卡训练入口
  hunyuan_adapter.py  # Hunyuan 加载、冻结、teacher/student MSE
  sla_adapter.py      # Dense attention 到 MindIE-SD SLA 的运行时替换
sampling/             # 离线 Hunyuan VAE latent 采样线
tools/                # Flickr30k/COYO 数据准备与 cache 检查等独立工具
configs/
  sampling.yaml       # 离线采样配置
  train_sla.yaml      # 模型、cache 和 SLA 参数
  accelerate_zero3_16npu.yaml # 16 NPU ZeRO-3 NPU-resident 性能配置
  accelerate_zero3_16npu_offload.yaml # CPU-offload 低显存 fallback
scripts/
  sample.sh           # 离线采样启动脚本
  train_sla.sh        # 单 NPU / DDP / ZeRO-3 训练脚本
doc/
  testing_linux.md     # Linux 环境部署、上游下载与验收流程
```

`upstream/` 是本地依赖目录，不提交到 Git。三个上游仓库的下载命令和完整环境准备流程见 [doc/testing_linux.md](doc/testing_linux.md)。

## Attention 替换位置

Hunyuan 原始 attention 位于：

```text
upstream/HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
HunyuanImage3SDPAAttention.forward
```

`train/sla_adapter.py` 会识别该模块并替换为 `HunyuanImage3SLAAttention`。适配器保留原始 QKV projection、RoPE、QK norm 和 `repeat_kv`，只将原有 SDPA 调用换成 MindIE-SD `SparseLinearAttention`。

## 快速启动

先按照 [Linux 部署与测试指南](doc/testing_linux.md) 下载上游仓库、安装 CANN/PyTorch NPU/MindIE-SD 依赖。离线采样和训练是两条独立流程，详细命令见 [离线采样](doc/offline_sampling.md)、[训练](doc/training.md) 与可直接执行的 [服务器测试命令](doc/server_test_commands.md)。

修改 `configs/sampling.yaml`：

```yaml
model_path: /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil
source:
  manifest_path: datasets/flickr30k/metadata.jsonl
  image_root: datasets/flickr30k/flickr30k-images
```

生成训练 cache：

```bash
bash scripts/sample.sh configs/sampling.yaml --resume
bash scripts/verify_cache.sh data/cache
```

训练读取 `data/cache` 中的 `latent_z0` 和静态 Hunyuan condition，不读取图片，也不执行 VAE encode。

正式 recovery 建议使用官方8步 Dense trajectory，而不是随机 timestep latent。采集器会
执行 Stage-0 AR/CoT、官方 MeanFlow rollout，保存每步 `x_t/t/r/teacher prediction`，
并在写入前完成scheduler和Dense replay验证。详见
[官方 Dense 8-step 轨迹采集](doc/trajectory_sampling.md)。

### vLLM-Omni Badcase 测评

下载 i2i 输入图、下载 t2i/i2i baseline 并批量调用已部署的 vLLM-Omni 服务，见
[Badcase 批量测评](doc/badcase_evaluation.md)。
[Badcase T2I 训练验证与实时曲线](doc/badcase_training_validation.md)说明如何构建固定
Dense trajectory验证集、记录JSONL指标并直接生成PNG曲线。
[SLA + Attention/MoE LoRA训练](doc/attention_moe_lora_training.md)提供rank-64 Attention
LoRA与rank-8 routed-expert down-projection LoRA的训练、导出和vLLM部署流程。
[Guidance=1 SLA Block稀疏度标定](doc/block_profile_guidance1.md)使用20条Dense trajectory
统计pooled Q/K累计质量，用于筛选HunyuanImage3合理的top-k候选。

### Dense Attention 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_sla.sh configs/train_sla.yaml --stage dense --max-steps 1
```

### SLA Recovery 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

成功时日志会分别输出 `proj_l/qkv_delta/o_delta` 的梯度统计，并在
`results/training/qkvo-delta/` 写入仅包含可训练参数的 checkpoint。

### 16 NPU ZeRO-3 启动

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

ZeRO-3 切分参数、梯度和 optimizer state，不是 TP/SP。ZeRO-3 checkpoint 保存为目录并排除冻结的 80B 基础参数；当前已在 16 张 910C A3 上完成 forward、loss、backward、optimizer step 和 checkpoint one-step 实机验收。

默认 profile 将 parameter/optimizer shard 常驻 NPU以减少 CPU 搬运。启动日志必须显示：

```text
deepspeed_offload_param_device=none deepspeed_offload_optimizer_device=none
```

若峰值 HBM 超过安全范围，切回 CPU-offload fallback：

```bash
ACCELERATE_CONFIG=configs/accelerate_zero3_16npu_offload.yaml \
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla
```

### 导出 vLLM-Omni SLA adapter

训练结束后将最新 ZeRO-3 checkpoint 中的 32 层 `proj_l` 和 QKV/O delta 导出为 safetensors：

```bash
bash scripts/export_sla_adapter.sh
```

也可以显式指定 step：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/qkvo-delta/sla-step-200 \
  results/adapters/qkvo-delta-step-200
```

输出包含 `adapter.safetensors`、`adapter_config.json` 和 `SHA256SUMS`，不包含基础模型或 optimizer。格式、检查方法及提供给 vLLM-Omni 的路径见 [SLA 权重导出](doc/sla_adapter_export.md)。

## 当前训练参数

SLA 阶段冻结原始 Hunyuan 参数。默认主实验训练：

```text
...self_attn.sla.proj_l.weight
...self_attn.sla.proj_l.bias
...self_attn.qkv_delta.weight
...self_attn.o_delta.weight
```

原始 `qkv_proj/o_proj` 不被修改，Dense teacher 始终使用冻结基座；student 使用
`base + delta`。训练强制使用带 `dQ/dK/dV` 的 Triton SLA，部署仍可使用 AscendC。
要复现实验 baseline，将 `trainable_components` 改为仅包含 `proj_l`，并使用独立
`output_dir`。可以直接运行：

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --trainable-components proj_l \
  --training-backend triton \
  --output-dir results/training/proj-only
```

默认 QKV/O 训练配置：`topk=0.125`、`BLKQ=128`、`BLKK=128`、BF16。
`head_dim=128, BLKQ=64, BLKK=128` 会在 910C Triton 编译时发生 UB overflow，
只保留给旧的 AscendC proj-only 推理 artifact。

性能实验支持独立覆盖 activation checkpoint 和每卡 micro batch。先分别测试：

```bash
# A: batch=1，关闭 checkpoint；峰值 HBM 超过 50-55GiB 时停止。
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla --max-steps 2 --micro-batch-size 1 --no-activation-checkpointing \
  --output-dir results/training/profile-no-checkpoint

# B: 保留 checkpoint，每卡 batch=2，全局 batch=32。
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla --max-steps 3 --micro-batch-size 2 --activation-checkpointing \
  --output-dir results/training/profile-batch2
```

batch=2 会按 packed sequence length 和分辨率精确分桶，不使用 padding。日志会打印
可用样本数和因桶不足被丢弃的样本数。只有 A/B 各自显存安全后才组合两项。

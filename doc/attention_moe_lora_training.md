# SLA + Attention/MoE LoRA 训练

该实验保留 SLA `proj_l` 全量训练，将原来的 1.342B QKV/O full-rank delta 替换为
Attention rank-64 LoRA，并为每个 routed MoE expert 的 `down_proj` 增加 rank-8 LoRA。

## 已确认的模型结构

服务器 checkpoint 配置为：

```text
hidden_size=4096
num_hidden_layers=32
num_attention_heads=32
num_key_value_heads=8
attention_head_dim=128
num_experts=64
moe_intermediate_size=[3072] * 32
moe_layer_num_skipped=0
moe_topk=[8] * 32
use_mixed_mlp_moe=True
num_shared_expert=[1] * 32
```

训练启动时还会审计加载后的实际模块：必须存在32个MoE层、每层64个routed expert，
且所有expert `down_proj`必须为`3072 -> 4096`。shared MLP、router和gate/up不包装。

## 参数量

```text
proj_l full train                 528,384
Attention QKV/O LoRA rank 64  37,748,736
MoE down_proj LoRA rank 8    117,440,512
------------------------------------------------
total                           155,717,632
```

LoRA使用A random、B ZeroInit和`scale=alpha/rank`，因此初始化时delta严格为零。
所有rank在模型构造前使用同一个全局seed，防止ZeRO-3下A初始化不一致。

## 16 NPU smoke

先确认离线训练和验证trajectory均已准备：

```bash
wc -l data/trajectories/manifest.jsonl
wc -l data/validation/badcase_t2i/trajectories/manifest.jsonl
```

运行一个step：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3

bash scripts/train_sla.sh configs/train_sla_attention_moe_lora.yaml \
  --max-steps 1 \
  --no-validation \
  --output-dir results/training/sla-attn-r64-moe-down-r8-smoke
```

启动日志必须包含：

```text
trainable_parameter_elements=155717632
sla_trainable_components=['proj_l', 'qkv_lora', 'o_lora']
moe_down_lora_geometry=MoELoRAGeometry(layers=32, experts_per_layer=64, ...)
```

step日志必须分别出现非零有限的：

```text
proj_l_grad_norm
attention_lora_grad_norm
moe_down_lora_grad_norm
```

首步LoRA A梯度为零是正常的，因为B从零开始；参数组整体必须有非零B梯度。

## 正式训练

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh \
  configs/train_sla_attention_moe_lora.yaml \
  --max-steps 250 \
  --output-dir results/training/sla-attn-r64-moe-down-r8
```

默认设置：

```text
16 NPU ZeRO-3
每卡 micro-batch=4
全局 batch=64
gradient accumulation=1
parameter offload=none
optimizer offload=none
activation checkpointing=on
proj_l LR=1e-4
Attention LoRA LR=1e-5
MoE down LoRA LR=3e-6
```

LoRA显著减少optimizer状态，但完整80B student forward、SLA backward、MoE dInput
backward和activation checkpoint重算仍存在，step时间不会按参数量降低8.6倍。

## 14 NPU 训练

14卡必须配套使用以下两个配置：

```text
configs/accelerate_zero3_14npu.yaml
configs/train_sla_attention_moe_lora_14npu.yaml
```

前者设置DeepSpeed world size，后者将固定验证集改为7条prompt：

```text
训练：14 ranks × micro-batch 4 = global batch 56
验证：7 prompts × 8 trajectory points = 56 points
```

不能继续使用16卡训练配置中的4条验证prompt，因为`32 % 14 != 0`，Accelerate会为
补齐分片而重复记录；训练代码会直接拒绝这种配置。

### 准备7条验证trajectory

已有验证manifest和trajectory不少于7条时可直接检查：

```bash
wc -l data/validation/badcase_t2i/trajectories/manifest.jsonl
find data/validation/badcase_t2i/trajectories/samples -name READY.json | wc -l
```

不足7条时重新构建固定manifest并采集。Stage-0和DiT采集各自使用8卡，采集进程退出、
释放NPU后再启动14卡训练：

```bash
python tools/build_badcase_validation_manifest.py \
  --cases datasets/test/badcase_t2i/cases.json \
  --output datasets/validation/badcase_t2i/prompts.jsonl \
  --limit 7

unset HUNYUAN_SLA_ADAPTER
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/sample_vllm_trajectories.sh \
  --phase stage0 \
  --config configs/vllm_badcase_validation_sampling.yaml \
  --manifest datasets/validation/badcase_t2i/prompts.jsonl \
  --limit 7 --resume

bash scripts/sample_vllm_trajectories.sh \
  --phase dit \
  --config configs/vllm_badcase_validation_sampling.yaml \
  --manifest data/validation/badcase_t2i/stage0_conditions/manifest.jsonl \
  --limit 7 --resume
```

### 14卡 smoke

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13
export TRAIN_PARALLEL=zero3
export ACCELERATE_CONFIG="$PWD/configs/accelerate_zero3_14npu.yaml"

bash scripts/train_sla.sh configs/train_sla_attention_moe_lora_14npu.yaml \
  --max-steps 1 \
  --no-validation \
  --output-dir results/training/sla-attn-r64-moe-down-r8-14npu-smoke
```

日志必须确认：

```text
world_size=14
deepspeed_train_micro_batch_size_per_gpu=4
deepspeed_offload_param_device=none
deepspeed_offload_optimizer_device=none
trainable_parameter_elements=155717632
```

### 14卡正式训练

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13
export TRAIN_PARALLEL=zero3
export ACCELERATE_CONFIG="$PWD/configs/accelerate_zero3_14npu.yaml"

bash scripts/train_sla.sh configs/train_sla_attention_moe_lora_14npu.yaml \
  --max-steps 250 \
  --output-dir results/training/sla-attn-r64-moe-down-r8-14npu
```

2000条prompt共有16000个trajectory point。global batch 56时完整遍历约需286步，最后
一个全局batch会发生补齐。默认250步只消费约14000个point，适合与16卡的250-step实验
按optimizer step对比，但不等于完整epoch。若论文比较要求相同见样本数，应以累计
`samples_seen = optimizer_steps × global_batch`为准，而不能只比较step数。

14卡比16卡每卡持有的ZeRO shard约多14.3%。默认仍关闭CPU offload并开启activation
checkpointing；先运行smoke并监控峰值HBM，建议保持在50-55GiB以内。

### 14卡断点恢复

恢复时必须继续使用14个进程、相同Accelerate配置和相同训练manifest：

```bash
export ACCELERATE_CONFIG="$PWD/configs/accelerate_zero3_14npu.yaml"

TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh \
  configs/train_sla_attention_moe_lora_14npu.yaml \
  --max-steps 250 \
  --resume-from results/training/sla-attn-r64-moe-down-r8-14npu/sla-step-125 \
  --output-dir results/training/sla-attn-r64-moe-down-r8-14npu
```

不能用14卡恢复16卡ZeRO-3 checkpoint，也不能反向混用；DeepSpeed optimizer shard数量
必须与保存checkpoint时一致。

## 导出 adapter v3

导出时必须传原始LoRA训练配置，否则无法恢复alpha/rank：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/sla-attn-r64-moe-down-r8/sla-step-250 \
  results/adapters/sla-attn-r64-moe-down-r8-step-250 \
  --config configs/train_sla_attention_moe_lora.yaml

python tools/inspect_sla_adapter.py \
  --adapter-dir results/adapters/sla-attn-r64-moe-down-r8-step-250
```

14卡checkpoint的LoRA几何相同，但应传对应训练配置并使用独立输出目录：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/sla-attn-r64-moe-down-r8-14npu/sla-step-250 \
  results/adapters/sla-attn-r64-moe-down-r8-14npu-step-250 \
  --config configs/train_sla_attention_moe_lora_14npu.yaml
```

预期：

```text
format_version=3
trained_components=[proj_l, qkv_lora, o_lora, moe_down_lora]
tensor_count=4288
parameter_count=155717632
```

## vLLM-Omni 推理

vLLM-Omni分支需要包含adapter v3 loader。loader在TP/EP分片前：

1. 对Attention LoRA计算`B @ A * alpha/rank`并保持fused GQA顺序。
2. 对每个HF expert `down_proj`逐个计算和合并delta。
3. 再交给现有FusedMoE loader进行TP/EP packing。

启动命令沿用QKVO SLA profile：

```bash
export HUNYUAN_SLA_ADAPTER=/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/sla-attn-r64-moe-down-r8-step-250
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve /mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_sla_qkvo.yaml \
  --enforce-eager --host 0.0.0.0 --port 8000
```

启动时需要对2048个expert逐个执行rank-8合并，模型加载会比旧QKV/O adapter慢，但
每次只物化一个完整delta，避免同时占用全部展开权重内存。

## 与旧实验的关系

旧full-delta checkpoint不能直接`--resume-from`到LoRA配置。该实验应从基础模型和
ZeroInit SLA重新训练，使用独立output目录，与原QKV/O full-rank baseline做同数据、
同step数、同验证集对比。

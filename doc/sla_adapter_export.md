# SLA 权重导出

训练 checkpoint 用于 DeepSpeed ZeRO-3 断点恢复，vLLM-Omni 不直接读取这些 rank
分片。训练结束后，由本仓库将 32 层 `proj_l` 与可选的 QKV/O delta 合并并导出为部署产物。

正式trajectory recovery训练使用 `results/training/trajectory-recovery/`，必须显式传入
checkpoint和输出目录；本页默认 `qkvo-delta/latest` 仅保留给旧实验。新命令见
[官方 Dense 8-step 轨迹采集](trajectory_sampling.md#导出新-adapter)。

## 默认导出最新 checkpoint

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
source /usr/local/Ascend/ascend-toolkit/set_env.sh

bash scripts/export_sla_adapter.sh
```

脚本读取：

```text
results/training/qkvo-delta/latest
```

并输出到：

```text
results/adapters/<latest-tag>/
```

## 导出指定 step

推荐正式交付时显式指定 checkpoint：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/qkvo-delta/sla-step-200 \
  results/adapters/qkvo-delta-step-200
```

覆盖已有导出：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/qkvo-delta/sla-step-200 \
  results/adapters/qkvo-delta-step-200 \
  --force
```

导出只使用 CPU，但 ZeRO-3 合并需要训练环境中的 DeepSpeed。转换 FP32 master weights
可能读取 `optim_states.pt`，因此验证导出成功前不要删除原始 checkpoint。

## 产物

```text
results/adapters/qkvo-delta-step-200/
├── adapter.safetensors
├── adapter_config.json
└── SHA256SUMS
```

v2 `adapter.safetensors` 默认包含：

```text
layers.0.sla.proj_l.weight
layers.0.sla.proj_l.bias
layers.0.qkv_delta.weight
layers.0.o_delta.weight
...
layers.31.sla.proj_l.weight
layers.31.sla.proj_l.bias
layers.31.qkv_delta.weight
layers.31.o_delta.weight
```

共 128 tensors，约 1.343B 参数。`proj_l` 以 FP32 导出，QKV/O delta 保持 BF16，
产物约 2.7GB。基础 HunyuanImage3 权重、optimizer、VAE 和 MoE 权重不会进入 adapter。
旧的 proj-only checkpoint 仍会导出为只含 64 tensors 的兼容 artifact。

## 单独检查

```bash
python tools/inspect_sla_adapter.py \
  --adapter-dir results/adapters/qkvo-delta-step-200

cd results/adapters/qkvo-delta-step-200
sha256sum -c SHA256SUMS
```

成功时必须显示：

```text
"valid": true
"tensor_count": 128
"dtype": ["torch.bfloat16", "torch.float32"]
```

## 提供给 vLLM-Omni

同一服务器可以直接使用绝对路径：

```text
/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/qkvo-delta-step-200/adapter.safetensors
```

vLLM-Omni 同时读取相邻的 `adapter_config.json`，并校验 `adapter_sha256`、层数、
shape 和参数数量。v2 QKV/O delta 会在 vLLM 做 TP 分片前逐层叠加到基础权重，
无需复制或合并完整 80B checkpoint。vLLM 仍从以下目录单独加载基础模型：

```text
/mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil
```

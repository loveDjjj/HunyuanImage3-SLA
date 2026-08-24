# SLA 权重导出

训练 checkpoint 用于 DeepSpeed ZeRO-3 断点恢复，vLLM-Omni 不直接读取这些 rank
分片。训练结束后，由本仓库将 32 层 `proj_l.weight/bias` 合并并导出为部署产物。

## 默认导出最新 checkpoint

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
source /usr/local/Ascend/ascend-toolkit/set_env.sh

bash scripts/export_sla_adapter.sh
```

脚本读取：

```text
results/training/default/latest
```

并输出到：

```text
results/adapters/<latest-tag>/
```

## 导出指定 step

推荐正式交付时显式指定 checkpoint：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/default/sla-step-200 \
  results/adapters/sla-step-200
```

覆盖已有导出：

```bash
bash scripts/export_sla_adapter.sh \
  results/training/default/sla-step-200 \
  results/adapters/sla-step-200 \
  --force
```

导出只使用 CPU，但 ZeRO-3 合并需要训练环境中的 DeepSpeed。转换 FP32 master weights
可能读取 `optim_states.pt`，因此验证导出成功前不要删除原始 checkpoint。

## 产物

```text
results/adapters/sla-step-200/
├── adapter.safetensors
├── adapter_config.json
└── SHA256SUMS
```

`adapter.safetensors` 固定包含：

```text
layers.0.sla.proj_l.weight
layers.0.sla.proj_l.bias
...
layers.31.sla.proj_l.weight
layers.31.sla.proj_l.bias
```

共 64 tensors、528384 个 FP32 参数。基础 HunyuanImage3 权重、optimizer、VAE 和
MoE 权重不会进入 adapter。

## 单独检查

```bash
python tools/inspect_sla_adapter.py \
  --adapter-dir results/adapters/sla-step-200

cd results/adapters/sla-step-200
sha256sum -c SHA256SUMS
```

成功时必须显示：

```text
"valid": true
"tensor_count": 64
"parameter_count": 528384
"dtype": ["torch.float32"]
```

## 提供给 vLLM-Omni

同一服务器可以直接使用绝对路径：

```text
/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/sla-step-200/adapter.safetensors
```

vLLM-Omni 同时读取相邻的 `adapter_config.json`，并校验 `adapter_sha256`、层数、
shape 和参数数量。vLLM 仍从以下目录单独加载基础模型：

```text
/mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil
```

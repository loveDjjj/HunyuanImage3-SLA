# SLA Recovery 训练

训练线只读取已经验证的 `data/cache`。每步从 `latent_z0` 生成新的 flow-matching `x_t`，以冻结 Dense Hunyuan 为 teacher，以替换 SLA attention 的 Hunyuan 为 student，优化二者 `diffusion_prediction` 的 MSE。

Instruct-Distil 使用 CFG distillation 和 MeanFlow。`guidance_index`、`timesteps_r_index` 属于离线保存的静态 token 位置；对应的 guidance 数值和 `r <= t` timestep 在每个训练 step 动态构造。默认 guidance scale 与官方 checkpoint 一致为 `2.5`，送入模型的 embedding 标量为 `2500`。

## 单 NPU 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# Dense forward/backward 基线
bash scripts/train_dense.sh configs/train_sla.yaml --max-steps 1

# SLA recovery forward/backward/optimizer/checkpoint
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

训练会拒绝未验证的 cache。成功日志必须包含 `finite_grad=True`，checkpoint 写入 `results/training/default/`。

## 16 NPU ZeRO-3

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla
```

该命令读取 `configs/accelerate_zero3_16npu.yaml`，使用 Accelerate + DeepSpeed ZeRO-3 切分模型参数、梯度和 optimizer state。它不切分 attention head、序列或 MoE expert，因此不是 TP、SP 或 EP。当前实现已经完成代码接入，但尚未在 16 张 910C A3 上完成 one-step 验收。

`save_every_steps` 控制周期保存。普通单卡/DDP checkpoint 是 `.pt` 文件；ZeRO-3 checkpoint 是所有 rank 共同写入的目录，并排除冻结的 80B 基础参数。

## 中断恢复

cache 按 `manifest.jsonl` 固定顺序读取，checkpoint 保存已完成 step。因此恢复时训练会跳过已消费样本，并以同一 sample/step 随机种子生成 `x_t`：

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --max-steps 200 \
  --resume-from results/training/default/sla-step-100
```

ZeRO-3 恢复必须使用与保存 checkpoint 时相同的 NPU 数量、基础模型、cache 和 Accelerate 配置。单卡模式仍使用类似 `--resume-from .../sla-step-100.pt` 的文件路径。

当 `max_steps` 大于 2,000 时，把 `num_epochs` 设为足够大的值；每个 epoch 都会对同一份 `latent_z0` 重新采样 timestep 和噪声。

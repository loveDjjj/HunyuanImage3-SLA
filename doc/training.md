# SLA Recovery 训练

训练线只读取已经验证的 `data/cache`。每步从 `latent_z0` 生成新的 flow-matching `x_t`，以冻结 Dense Hunyuan 为 teacher，以替换 SLA attention 的 Hunyuan 为 student，优化二者 `diffusion_prediction` 的 MSE。

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

## 多 NPU

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla
```

这是未实机验证的 DDP 启动路径，不是模型并行方案。DDP 在每张卡复制完整模型，仅在单卡可容纳完整 Hunyuan checkpoint、Dense teacher 和 SLA student 时使用；当前不支持 SP、TP、EP、ZeRO。对于 80B checkpoint，应先完成模型并行训练适配，再进行多 NPU recovery。

checkpoint 保存可训练 SLA 参数、optimizer state 和已完成 step；训练 cache 固定顺序读取，不依赖在线 VAE 采样。

## 中断恢复

cache 按 `manifest.jsonl` 固定顺序读取，checkpoint 保存已完成 step。因此恢复时训练会跳过已消费样本，并以同一 sample/step 随机种子生成 `x_t`：

```bash
bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --max-steps 200 \
  --resume-from results/training/default/sla-step-100.pt
```

多卡恢复必须使用与保存 checkpoint 时相同的 NPU 数量和配置。

当 `max_steps` 大于 2,000 时，把 `num_epochs` 设为足够大的值；每个 epoch 都会对同一份 `latent_z0` 重新采样 timestep 和噪声。

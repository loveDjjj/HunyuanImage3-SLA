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

该命令读取 `configs/accelerate_zero3_16npu.yaml`，使用 Accelerate + DeepSpeed ZeRO-3 切分模型参数、梯度和 optimizer state。它不切分 attention head、序列或 MoE expert，因此不是 TP、SP 或 EP。当前实现已在 16 张 910C A3 上完成 one-step 验收：loss 有限、32 层 `proj_l` 梯度完整且有限、optimizer step 和 ZeRO checkpoint 均成功。

针对 64 GiB NPU，默认将 ZeRO-3 parameter/optimizer state offload 到 CPU，并对 32 个 Hunyuan decoder layer 启用 activation checkpointing。Dense teacher 在 `no_grad` 下不重算；SLA student 在 backward 期间逐层重算，以训练时间换取激活显存。节点需具备足够主机内存。

`save_every_steps` 控制周期保存。普通单卡/DDP checkpoint 是 `.pt` 文件；ZeRO-3 checkpoint 是所有 rank 共同写入的目录，并排除冻结的 80B 基础参数。

16 卡 ZeRO-3 checkpoint 每个 step 会生成 32 个主要分片文件：

- `zero_pp_rank_<rank>_mp_rank_00_model_states.pt`：该 rank 的模型 checkpoint 元数据、client state 和训练参数状态。冻结的 80B 基础参数已排除，恢复时仍从 `model_path` 重新加载。
- `bf16_zero_pp_rank_<rank>_mp_rank_00_optim_states.pt`：该 rank 的 ZeRO 优化器分片，包括 FP32 master parameter、Adam moments 和分片信息。
- `latest`：记录默认恢复的最新 tag，例如 `sla-step-1`。
- `zero_to_fp32.py`：DeepSpeed 自动生成的离线合并辅助脚本，不是额外一份模型权重。

同一 checkpoint 目录中的 16 个 model shard 和 16 个 optimizer shard 是一个整体。需要继续训练时不能只保留 rank 0 文件，也不能混用不同 step 的分片。检查体积和 tag：

```bash
du -sh results/training/default/sla-step-1
du -h results/training/default/sla-step-1/* | sort -h
cat results/training/default/latest
```

保存前，每个 rank 都会创建相同的 `sla-step-N` tag 目录并执行 barrier，避免 DeepSpeed 多 rank 同时写入时出现 `Parent directory ... does not exist`。配置中的相对 `output_dir` 会统一解析到仓库根目录。

`scripts/train_sla.sh` 和 `scripts/train_dense.sh` 会在启动时打印 `training_log=...`，完整 stdout/stderr 同时写入 `logs/training/<时间>.log`。训练进度条只由主 rank 输出，避免 16 个进度条互相覆盖。

## 中断恢复

cache 按 `manifest.jsonl` 固定顺序读取，checkpoint 保存已完成 step。因此恢复时训练会跳过已消费样本，并以同一 sample/step 随机种子生成 `x_t`：

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --max-steps 200 \
  --resume-from results/training/default/sla-step-100
```

ZeRO-3 恢复必须使用与保存 checkpoint 时相同的 NPU 数量、基础模型、cache 和 Accelerate 配置。单卡模式仍使用类似 `--resume-from .../sla-step-100.pt` 的文件路径。

`max_steps` 是实际停止目标，`num_epochs` 只作为最小 epoch 数。训练入口会根据 Accelerate 切分后的 `len(dataloader)` 自动扩展有效 epoch 数。例如 2,000 条 cache 使用 16 rank 时，每个 rank 每个 epoch 有 125 个 batch；`max_steps=200` 会自动使用 2 个 epoch。每次重复使用同一份 `latent_z0` 时都会按 epoch、step 和 sample id 重新采样 timestep 与噪声。

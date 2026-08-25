# SLA Recovery 训练

训练线只读取已经验证的 `data/cache`。每步从 `latent_z0` 生成新的 flow-matching `x_t`，以冻结 Dense Hunyuan 为 teacher，以替换 SLA attention 的 Hunyuan 为 student，优化二者 `diffusion_prediction` 的 MSE。

Instruct-Distil 使用 CFG distillation 和 MeanFlow。`guidance_index`、`timesteps_r_index` 属于离线保存的静态 token 位置；对应的 guidance 数值和 `r <= t` timestep 在每个训练 step 动态构造。默认 guidance scale 与官方 checkpoint 一致为 `2.5`，送入模型的 embedding 标量为 `2500`。

QKV/O adaptation 强制使用 Triton SLA，并使用 `head_dim=128, BLKQ=128,
BLKK=128`。不要改回 `BLKQ=64`；Triton 的 `128/64/128` kernel 在 910C 上会因
UB overflow 编译失败。

## 单 NPU 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# Dense forward/backward 基线
bash scripts/train_dense.sh configs/train_sla.yaml --max-steps 1

# SLA recovery forward/backward/optimizer/checkpoint
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

训练会拒绝未验证的 cache。成功日志必须包含 `finite_grad=True`，并分别报告
`proj_l/qkv_delta/o_delta` 的非零有限梯度。checkpoint 写入
`results/training/qkvo-delta/`。

## 16 NPU ZeRO-3

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla
```

该命令读取 `configs/accelerate_zero3_16npu.yaml`，使用 Accelerate + DeepSpeed ZeRO-3
切分模型参数、梯度和 optimizer state。它不切分 attention head、序列或 MoE expert，
因此不是 TP、SP 或 EP。此前 16 张 910C A3 的 one-step 结果验证的是 0.53M
proj-only baseline；新的 1.343B QKV/O delta 配置已完成 CPU-offload one-step，
NPU-resident 性能 profile 仍需重新执行显存和稳定性验收。

默认 `accelerate_zero3_16npu.yaml` 将 ZeRO-3 parameter 和 optimizer shard 常驻 NPU，
减少 CPU/NPU 参数搬运并提高 AICore duty cycle；32 个 decoder layer 仍启用 activation
checkpointing。建议峰值 HBM 不超过 50-55GiB。若发生 OOM，使用：

```bash
ACCELERATE_CONFIG=configs/accelerate_zero3_16npu_offload.yaml \
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla
```

fallback 会将 parameter/optimizer state offload 到 CPU，以吞吐换显存。

MindIE-SD 原生将 `proj_l` 初始化为 FP32，而 QKV/O delta 跟随 BF16 基座。NPU-resident
ZeRO-3 flat buffer 不接受混合 dtype，因此训练适配层会在 `accelerator.prepare()` 前
统一全部 trainable parameter 为 BF16；Adam 的 FP32 master state 和导出的 FP32
`proj_l` 不受影响。启动日志应显示 `trainable_parameter_dtypes=['torch.bfloat16']`。

默认只由 rank 0 输出 `dense_teacher_forward`、`sla_student_forward`、`backward` 和
`optimizer` 阶段标记。上游运行时若输出无标签的连续点，可以通过相邻阶段标记判断
它来自 teacher、student 还是 activation checkpoint backward 重算。

`save_every_steps` 控制周期保存。普通单卡/DDP checkpoint 是 `.pt` 文件；ZeRO-3 checkpoint 是所有 rank 共同写入的目录，并排除冻结的 80B 基础参数。

16 卡 ZeRO-3 checkpoint 每个 step 会生成 32 个主要分片文件：

- `zero_pp_rank_<rank>_mp_rank_00_model_states.pt`：该 rank 的模型 checkpoint 元数据、client state 和训练参数状态。冻结的 80B 基础参数已排除，恢复时仍从 `model_path` 重新加载。
- `bf16_zero_pp_rank_<rank>_mp_rank_00_optim_states.pt`：该 rank 的 ZeRO 优化器分片，包括 FP32 master parameter、Adam moments 和分片信息。
- `latest`：记录默认恢复的最新 tag，例如 `sla-step-1`。
- `zero_to_fp32.py`：DeepSpeed 自动生成的离线合并辅助脚本，不是额外一份模型权重。

同一 checkpoint 目录中的 16 个 model shard 和 16 个 optimizer shard 是一个整体。需要继续训练时不能只保留 rank 0 文件，也不能混用不同 step 的分片。检查体积和 tag：

```bash
du -sh results/training/qkvo-delta/sla-step-1
du -h results/training/qkvo-delta/sla-step-1/* | sort -h
cat results/training/qkvo-delta/latest
```

保存前，每个 rank 都会创建相同的 `sla-step-N` tag 目录并执行 barrier，避免 DeepSpeed 多 rank 同时写入时出现 `Parent directory ... does not exist`。配置中的相对 `output_dir` 会统一解析到仓库根目录。

`scripts/train_sla.sh` 和 `scripts/train_dense.sh` 会在启动时打印 `training_log=...`，完整 stdout/stderr 同时写入 `logs/training/<时间>.log`。训练进度条只由主 rank 输出，避免 16 个进度条互相覆盖。

## 中断恢复

cache 按 `manifest.jsonl` 固定顺序读取，checkpoint 保存已完成 step。因此恢复时训练会跳过已消费样本，并以同一 sample/step 随机种子生成 `x_t`：

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla.yaml \
  --stage sla \
  --max-steps 200 \
  --resume-from results/training/qkvo-delta/sla-step-100
```

ZeRO-3 恢复必须使用与保存 checkpoint 时相同的 NPU 数量、基础模型、cache 和 Accelerate 配置。单卡模式仍使用类似 `--resume-from .../sla-step-100.pt` 的文件路径。

`max_steps` 是实际停止目标，`num_epochs` 只作为最小 epoch 数。训练入口会根据 Accelerate 切分后的 `len(dataloader)` 自动扩展有效 epoch 数。例如 2,000 条 cache 使用 16 rank 时，每个 rank 每个 epoch 有 125 个 batch；`max_steps=200` 会自动使用 2 个 epoch。每次重复使用同一份 `latent_z0` 时都会按 epoch、step 和 sample id 重新采样 timestep 与噪声。

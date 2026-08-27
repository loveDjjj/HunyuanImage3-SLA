# SLA Recovery 训练

正式训练读取已经验证的 `data/trajectories`。每个prompt包含官方Dense
HunyuanImage3-Instruct-Distil完整8步MeanFlow轨迹；训练直接读取缓存的
`x_t/t/r/condition/teacher_diffusion_prediction`，不再在线运行Dense teacher。

旧 `data/cache` 的随机timestep路径仅保留作历史baseline，不应用于正式QKV/O adapter
或部署质量结论。

trajectory保存Stage-0 AR/CoT/recaption后的真实condition、exact mixed causal/full
mask、guidance `2500`、官方`t/r`和FP32 teacher prediction。Recovery loss固定使用
FP32 MSE。采集和硬验证见[官方 Dense 8-step 轨迹采集](trajectory_sampling.md)。
固定 badcase 验证集、实时 JSONL/PNG 曲线和 checkpoint 图片对比见
[Badcase T2I 训练验证与实时曲线](badcase_training_validation.md)。
rank-64 Attention LoRA与rank-8 MoE expert down-projection LoRA实验见
[SLA + Attention/MoE LoRA训练](attention_moe_lora_training.md)。

QKV/O adaptation 强制使用 Triton SLA，并使用 `head_dim=128, BLKQ=128,
BLKK=128`。不要改回 `BLKQ=64`；Triton 的 `128/64/128` kernel 在 910C 上会因
UB overflow 编译失败。

## 16卡 trajectory smoke

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3

wc -l data/trajectories/manifest.jsonl
bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 5 --no-validation \
  --output-dir results/training/trajectory-smoke
```

训练会拒绝未验证的trajectory。成功日志必须显示 `phase=cached_dense_teacher`、
`finite_grad=True`，并分别报告
`proj_l/qkv_delta/o_delta` 的非零有限梯度。checkpoint 写入
`results/training/trajectory-smoke/`。

## 16 NPU ZeRO-3

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 250 \
  --output-dir results/training/trajectory-recovery
```

该命令读取 `configs/accelerate_zero3_16npu.yaml`，使用 Accelerate + DeepSpeed ZeRO-3
切分模型参数、梯度和 optimizer state。它不切分 attention head、序列或 MoE expert，
因此不是 TP、SP 或 EP。2000个prompt对应16000个trajectory point；默认每卡batch4、
全局batch64，一个完整epoch为250 optimizer steps。Dataset按exact condition布局分桶，
不会通过padding改变SLA block或teacher语义。

默认 `accelerate_zero3_16npu.yaml` 将 ZeRO-3 parameter 和 optimizer shard 常驻 NPU，
减少 CPU/NPU 参数搬运并提高 AICore duty cycle；32 个 decoder layer 仍启用 activation
checkpointing。建议峰值 HBM 不超过 50-55GiB。若发生 OOM，使用：

```bash
ACCELERATE_CONFIG=configs/accelerate_zero3_16npu_offload.yaml \
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 250 \
  --output-dir results/training/trajectory-recovery
```

fallback 会将 parameter/optimizer state offload 到 CPU，以吞吐换显存。

MindIE-SD 原生将 `proj_l` 初始化为 FP32，而 QKV/O delta 跟随 BF16 基座。NPU-resident
ZeRO-3 flat buffer 不接受混合 dtype，因此训练适配层会在 `accelerator.prepare()` 前
统一全部 trainable parameter 为 BF16；Adam 的 FP32 master state 和导出的 FP32
`proj_l` 不受影响。启动日志应显示 `trainable_parameter_dtypes=['torch.bfloat16']`。

默认只由 rank 0 输出阶段标记。trajectory 缓存训练应显示
`cached_dense_teacher`、`sla_student_forward`、`backward` 和 `optimizer`；旧的在线
teacher 路径才会显示 `dense_teacher_forward`。上游运行时若输出无标签的连续点，
可以通过相邻阶段标记判断它来自 student 还是 activation checkpoint backward 重算。

正式配置每步向 `output_dir/metrics/metrics.jsonl` 追加全局 loss、各参数组梯度范数、
step 耗时、吞吐和峰值 NPU 内存；每5步原子刷新 `training_metrics.png`。每25步在4条
固定 `badcase_t2i` prompt的32个Dense trajectory point上计算验证指标。该验证缓存需在
训练前准备；临时 smoke 可传 `--no-validation`。

## Activation checkpoint 性能实验

trajectory recovery默认每卡batch4，并且只组合exact condition布局相同的记录。
关闭checkpoint可省去student backward逐层重算，但batch4下可能显著增加激活峰值：

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 2 --micro-batch-size 4 --no-activation-checkpointing \
  --output-dir results/training/trajectory-no-checkpoint-profile
```

峰值HBM建议不超过50-55GiB；超过时恢复默认checkpoint配置。

`save_every_steps` 控制周期保存。普通单卡/DDP checkpoint 是 `.pt` 文件；ZeRO-3 checkpoint 是所有 rank 共同写入的目录，并排除冻结的 80B 基础参数。

16 卡 ZeRO-3 checkpoint 每个 step 会生成 32 个主要分片文件：

- `zero_pp_rank_<rank>_mp_rank_00_model_states.pt`：该 rank 的模型 checkpoint 元数据、client state 和训练参数状态。冻结的 80B 基础参数已排除，恢复时仍从 `model_path` 重新加载。
- `bf16_zero_pp_rank_<rank>_mp_rank_00_optim_states.pt`：该 rank 的 ZeRO 优化器分片，包括 FP32 master parameter、Adam moments 和分片信息。
- `latest`：记录默认恢复的最新 tag，例如 `sla-step-1`。
- `zero_to_fp32.py`：DeepSpeed 自动生成的离线合并辅助脚本，不是额外一份模型权重。

同一 checkpoint 目录中的 16 个 model shard 和 16 个 optimizer shard 是一个整体。需要继续训练时不能只保留 rank 0 文件，也不能混用不同 step 的分片。检查体积和 tag：

```bash
du -sh results/training/trajectory-recovery/sla-step-250
du -h results/training/trajectory-recovery/sla-step-250/* | sort -h
cat results/training/trajectory-recovery/latest
```

保存前，每个 rank 都会创建相同的 `sla-step-N` tag 目录并执行 barrier，避免 DeepSpeed 多 rank 同时写入时出现 `Parent directory ... does not exist`。配置中的相对 `output_dir` 会统一解析到仓库根目录。

`scripts/train_sla.sh` 和 `scripts/train_dense.sh` 会在启动时打印 `training_log=...`，完整 stdout/stderr 同时写入 `logs/training/<时间>.log`。训练进度条只由主 rank 输出，避免 16 个进度条互相覆盖。

## 中断恢复

trajectory manifest和每个样本内部8步顺序固定，checkpoint保存已完成step。恢复命令：

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla \
  --max-steps 250 \
  --resume-from results/training/trajectory-recovery/sla-step-125 \
  --output-dir results/training/trajectory-recovery
```

ZeRO-3恢复必须使用相同NPU数量、基础模型、trajectory manifest和Accelerate配置。

`max_steps`是实际停止目标，`num_epochs`只作为最小epoch数。2000条prompt产生16000
trajectory points，16 rank、每卡batch4时每个完整epoch为250 optimizer steps。

## 导出和部署

```bash
bash scripts/export_sla_adapter.sh \
  results/training/trajectory-recovery/sla-step-250 \
  results/adapters/trajectory-recovery-step-250

python tools/inspect_sla_adapter.py \
  --adapter-dir results/adapters/trajectory-recovery-step-250
```

部署前更新vLLM-Omni适配分支，确保包含MeanFlow timestep-r和global-prefix SLA修复；
然后将 `HUNYUAN_SLA_ADAPTER` 指向该新目录。旧随机timestep adapter不再兼容正式链路。

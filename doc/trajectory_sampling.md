# 官方 Dense 8-step 轨迹采集

需要使用vLLM-Omni TP/EP加速大规模采集时，见
[vLLM-Omni Dense Teacher 轨迹采集](vllm_trajectory_sampling.md)。本页保留为官方HF
参考实现和数值对照基线。

本流程直接调用 Tencent HunyuanImage-3.0-Instruct-Distil 的官方 `generate_image()`，
保留 Stage-0 AR/CoT/recaption、官方8步 MeanFlow scheduler、exact mixed attention mask
和每一步 Dense diffusion prediction。轨迹用于 SLA recovery 时不再在线运行 Dense teacher。

## 文件格式

```text
data/trajectories/
├── manifest.jsonl
└── samples/sample_000001/
    ├── metadata.json
    ├── trajectory.safetensors
    ├── final.png              # save_final_image=true 时存在
    └── READY.json
```

核心 tensor：

```text
latents                    [9,32,64,64] FP32
teacher_predictions        [8,32,64,64] FP32
timesteps                  [8] FP32
timesteps_r                [8] FP32
input_ids                  [1,L] INT64
position_ids               [1,L] INT64
image_mask                 [1,L] BOOL
attention_mask_packed      [ceil(L*L/8)] UINT8
timesteps_index            [1,Kt] INT64
guidance_index             [1,Kg] INT64
timesteps_r_index          [1,Kr] INT64
gen_timestep_scatter_index [1,Ks] INT64
guidance                   [1] BF16
ar_generated_token_ids     [N] INT64
```

`latents[i]` 是第i步的 `x_t`，`latents[i+1]` 是该步scheduler输出。每个样本只保存
一份condition和bit-packed exact mask。1024分辨率不含PNG约10.5-11MiB，2000条约
21-22GiB。

## 准备2000条 Prompt Manifest

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
mkdir -p datasets

jq -c \
  '{id: (.id | tostring), prompt: .caption, seed: 42}' \
  datasets/flickr30k/metadata.jsonl \
  | head -n 2000 \
  > datasets/trajectory_prompts.jsonl

head -n 2 datasets/trajectory_prompts.jsonl
wc -l datasets/trajectory_prompts.jsonl
```

每行必须包含 `id/prompt/seed`：

```json
{"id":"10000","prompt":"A classroom full of students with laptop computers.","seed":42}
```

## 16卡单prompt硬验证

优先使用NPU-resident ZeRO-3。所有16个rank共同执行同一个80B官方rollout，并非并行
采16个prompt；只有rank0写文件。CPU parameter offload会在Stage-0逐token AR期间反复
搬运参数，速度可能慢一个数量级，不适合作为常规采集配置。

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACCELERATE_CONFIG="$PWD/configs/accelerate_zero3_16npu.yaml"

bash scripts/sample_trajectories.sh \
  --config configs/trajectory_sampling.yaml \
  --manifest datasets/trajectory_prompts.jsonl \
  --limit 1
```

假设第一条ID为10000：

```bash
python tools/inspect_trajectory.py \
  --sample-dir data/trajectories/samples/sample_10000
```

采集器在写 `READY.json` 前强制验证：

1. 恰好8个teacher prediction和9个连续latent。
2. `t/r` 与官方scheduler的 `timesteps/timesteps_full[1:]` 完全一致。
3. 使用teacher prediction连续scheduler重放，逐步与缓存latent完全一致。
4. 使用完整condition、exact mask和 `use_cache=False` 重算Dense forward，与官方
   KV-cache rollout prediction满足配置的FP32容差。

采集期间仅 rank 0 显示进度。`trajectory sampling` 是prompt总进度，`stage0 AR`显示
AR/CoT逐token进度；进入去噪后会显示 `dense rollout 0/8`，数值复验阶段显示
`dense replay 0/8`。其他rank的重复Transformers和NPU格式warning会被抑制。

检查结果必须包含 `valid=true`、8个prediction、9个latent、完全一致的`t/r`和
`scheduler_replay_max_abs=0`。

## 16卡完整采集和恢复

推荐的NPU-resident版本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACCELERATE_CONFIG="$PWD/configs/accelerate_zero3_16npu.yaml"

bash scripts/sample_trajectories.sh \
  --config configs/trajectory_sampling.yaml \
  --manifest datasets/trajectory_prompts.jsonl \
  --limit 2000 \
  --resume
```

只有NPU-resident发生OOM时才使用CPU-offload fallback：

```bash
export ACCELERATE_CONFIG="$PWD/configs/accelerate_zero3_16npu_offload.yaml"

bash scripts/sample_trajectories.sh \
  --config configs/trajectory_sampling.yaml \
  --manifest datasets/trajectory_prompts.jsonl \
  --limit 2000 \
  --resume
```

resident采集期间持续监控 `npu-smi info`，峰值建议不超过50-55GiB。OOM时切回
offload profile并保留 `--resume`，但Stage-0 AR会显著变慢。只有含合法 `READY.json`
的样本会被跳过，失败或replay验证未通过的样本不会进入总manifest。

## 使用离线teacher训练

2000个prompt各含8步，共16000个训练点；默认16卡、每卡batch4，全局batch64，
一个完整epoch为250 optimizer steps。Dataset按exact condition布局分桶，同一prompt
的8步天然可以组成两个batch4，不对attention mask做padding。先检查数据并跑smoke：

```bash
wc -l data/trajectories/manifest.jsonl
python tools/inspect_trajectory.py \
  --sample-dir data/trajectories/samples/sample_10000

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3

bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 5 \
  --output-dir results/training/trajectory-smoke
```

正式一epoch示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3

bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla \
  --max-steps 250 \
  --output-dir results/training/trajectory-recovery
```

日志应显示 `phase=cached_dense_teacher`，而不是 `phase=dense_teacher_forward`。
trajectory训练使用官方 mixed causal/full mask，recovery loss固定为FP32 MSE。

断点恢复：

```bash
TRAIN_PARALLEL=zero3 bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 250 \
  --resume-from results/training/trajectory-recovery/sla-step-125 \
  --output-dir results/training/trajectory-recovery
```

## 导出新 adapter

```bash
bash scripts/export_sla_adapter.sh \
  results/training/trajectory-recovery/sla-step-250 \
  results/adapters/trajectory-recovery-step-250

python tools/inspect_sla_adapter.py \
  --adapter-dir results/adapters/trajectory-recovery-step-250

(cd results/adapters/trajectory-recovery-step-250 && sha256sum -c SHA256SUMS)
```

不要继续部署旧的随机timestep QKVO adapter。新部署路径：

```bash
export HUNYUAN_SLA_ADAPTER=/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/trajectory-recovery-step-250
```

# 官方 Dense 8-step 轨迹采集

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

## 单prompt实机采集

在8张 Ascend NPU、训练环境中运行：

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/sample_trajectories.sh \
  --prompt "A cinematic portrait of an astronaut standing in a futuristic greenhouse, highly detailed" \
  --seed 42 \
  --sample-id 000001
```

默认使用 `configs/accelerate_zero3_trajectory_8npu.yaml` 的 ZeRO-3 CPU parameter
offload。所有rank共同执行同一个官方rollout，仅rank0原子写文件。

检查：

```bash
python tools/inspect_trajectory.py \
  --sample-dir data/trajectories/samples/sample_000001
```

采集器在写 `READY.json` 前强制验证：

1. 恰好8个teacher prediction和9个连续latent。
2. `t/r` 与官方scheduler的 `timesteps/timesteps_full[1:]` 完全一致。
3. 使用teacher prediction连续scheduler重放，逐步与缓存latent完全一致。
4. 使用完整condition、exact mask和 `use_cache=False` 重算Dense forward，与官方
   KV-cache rollout prediction满足配置的FP32容差。

## JSONL批量采集和恢复

manifest每行：

```json
{"id":"000001","prompt":"...","seed":42}
```

```bash
bash scripts/sample_trajectories.sh \
  --manifest datasets/trajectory_prompts.jsonl \
  --limit 10 \
  --resume
```

只有含合法 `READY.json` 的样本会被跳过。失败样本不会进入总 `manifest.jsonl`。

## 使用离线teacher训练

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3

bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla \
  --max-steps 200 \
  --output-dir results/training/trajectory-recovery
```

日志应显示 `phase=cached_dense_teacher`，而不是 `phase=dense_teacher_forward`。
trajectory训练使用官方 mixed causal/full mask，recovery loss固定为FP32 MSE。

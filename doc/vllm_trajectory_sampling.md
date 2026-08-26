# vLLM-Omni Dense Teacher 轨迹采集

该流程使用vLLM-Omni的Hunyuan Stage-0 TP/FusedMoE和Dense DiT TP+EP，输出格式与
`doc/trajectory_sampling.md`中的官方HF采集完全相同。训练继续读取
`data/trajectories`，无需修改dataset或训练命令。

## 前置条件

服务器需要同时更新两个仓库，并在vLLM容器内使用editable安装：

```bash
cd /mnt/share/r50063443/vllm-omni
git pull
python -m pip install -e . --no-deps

cd /mnt/share/r50063443/HunyuanImage3-SLA
git pull
```

确认配置中的路径：

```bash
sed -n '1,120p' configs/vllm_trajectory_sampling.yaml
```

Teacher采集必须满足：

```text
Dense TORCH_SDPA
8-step MeanFlow
guidance_scale=2.5
temperature=0
不加载HUNYUAN_SLA_ADAPTER
不启用Cache-DiT/TeaCache/Taylor cache
不量化
```

## 8卡串行采集

8卡不能同时常驻AR和80B DiT，因此先批量生成Stage-0 condition，进程退出并释放HBM，
再启动DiT trajectory采集。

### 1. Stage-0 smoke

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
unset HUNYUAN_SLA_ADAPTER
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_OMNI_REPO=/mnt/share/r50063443/vllm-omni

bash scripts/sample_vllm_trajectories.sh \
  --phase stage0 \
  --config configs/vllm_trajectory_sampling.yaml \
  --manifest datasets/trajectory_prompts.jsonl \
  --limit 1 \
  --resume
```

检查：

```bash
cat data/stage0_conditions/manifest.jsonl
cat data/stage0_conditions/samples/sample_10000.json
```

每条记录必须包含`prompt_token_ids/generated_token_ids/cot_text`。Stage-0完成后脚本会
关闭Omni engine；使用`npu-smi info`确认8个AR worker已经退出，再运行DiT。

Stage-0使用generator逐请求返回：一个请求完成后立即原子写入对应
`samples/sample_<id>.json`，不会等待整批结束。DiT同样逐样本返回，并使用单线程有界
CPU writer与后续NPU计算重叠；最多保留两个待写trajectory。`READY.json`最后写入，
因此中断后`--resume`只会重做未完成样本。

### 2. DiT smoke

```bash
unset HUNYUAN_SLA_ADAPTER
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/sample_vllm_trajectories.sh \
  --phase dit \
  --config configs/vllm_trajectory_sampling.yaml \
  --manifest data/stage0_conditions/manifest.jsonl \
  --limit 1 \
  --resume
```

```bash
python tools/inspect_trajectory.py \
  --sample-dir data/trajectories/samples/sample_10000
```

必须显示9个FP32 latent、8个FP32 teacher prediction和8个`t/r`。
vLLM step-execution会在每次scheduler更新后将下一步latent转回BF16；artifact保存的是
这些真实DiT输入的FP32副本，metadata中的`scheduler_latent_dtype=bfloat16`用于按相同
cast规则执行scheduler replay。

### 3. 完整8卡任务

```bash
bash scripts/sample_vllm_trajectories.sh \
  --phase stage0 \
  --config configs/vllm_trajectory_sampling.yaml \
  --manifest datasets/trajectory_prompts.jsonl \
  --limit 2000 \
  --resume

# 等Stage-0进程完全退出后执行：
bash scripts/sample_vllm_trajectories.sh \
  --phase dit \
  --config configs/vllm_trajectory_sampling.yaml \
  --manifest data/stage0_conditions/manifest.jsonl \
  --limit 2000 \
  --resume
```

同一次DiT调用中的manifest记录必须使用相同seed。当前正式manifest统一使用42。

## 16卡流水线采集

16卡时AR使用0-7，DiT使用8-15，可以直接运行端到端pipeline：

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
unset HUNYUAN_SLA_ADAPTER
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_OMNI_REPO=/mnt/share/r50063443/vllm-omni

bash scripts/sample_vllm_trajectories.sh \
  --phase full \
  --config configs/vllm_trajectory_sampling.yaml \
  --manifest datasets/trajectory_prompts.jsonl \
  --limit 1 \
  --resume
```

单样本通过后改为`--limit 2000`。部署拓扑来自：

```text
vllm_omni/deploy/hunyuan_image_3_distil_trajectory_16npu.yaml
```

## 输出与训练

输出仍为：

```text
data/trajectories/
├── manifest.jsonl
└── samples/sample_<id>/
    ├── metadata.json
    ├── trajectory.safetensors
    └── READY.json
```

metadata会额外记录：

```text
teacher_backend=vllm-omni-dense
vllm_omni_commit
repository_commit
```

训练命令不变：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3

bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 5 \
  --output-dir results/training/vllm-trajectory-smoke
```

## 验收顺序

1. 先跑1条8卡Stage-0和DiT。
2. 对同一prompt运行现有HF sampler，比较Stage-0 token IDs。
3. 比较两侧`input_ids/attention mask/t/r`。
4. 比较8步teacher prediction的FP32误差和最终latent。
5. 单请求验收后再增加vLLM `max_num_seqs`，不能先扩大并发。

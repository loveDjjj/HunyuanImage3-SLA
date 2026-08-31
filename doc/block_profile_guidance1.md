# Guidance=1 SLA Block稀疏度标定

本流程在独立分支`feat/sla-block-profile-guidance1`上运行，只分析Dense HunyuanImage3
的Q/K block分布，不训练模型、不修改Triton kernel，也不保存完整Attention矩阵。

## 统计定义

实现严格复现当前MindIE-SD router的预处理：

```text
K' = K - Mean(K)
Q_block = MeanPool(Q, 128)
K_block = MeanPool(K', 128)
score = Q_block @ K_block^T
```

Top-K排序与实际kernel一致。为了衡量累计信息质量，按SLA论文定义计算：

```text
PooledMass = Softmax(score / sqrt(head_dim))
```

对每个prompt、MeanFlow step、layer、head和image query block统计：

- 候选top-k保留的累计pooled mass recall。
- recall的mean、P10和P05。
- 覆盖90%/95%/99% pooled mass所需block比例的mean、P90和P95。
- 推荐top-k、逐layer和逐MeanFlow step分布。

该指标是router proxy mass，不是token-level Dense Attention真实概率质量；用于筛选候选
稀疏度，最终仍需结合20条自由rollout、图片质量和vLLM延迟确认。

## 1. 构建20条固定Prompt

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA

python tools/build_badcase_validation_manifest.py \
  --cases datasets/test/badcase_t2i/cases.json \
  --output datasets/block_profile_guidance1/prompts.jsonl \
  --limit 20
```

## 2. 采集Guidance=1 Stage-0 Condition

```bash
unset HUNYUAN_SLA_ADAPTER
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/sample_vllm_trajectories.sh \
  --phase stage0 \
  --config configs/vllm_block_profile_guidance1.yaml \
  --manifest datasets/block_profile_guidance1/prompts.jsonl \
  --limit 20 --resume
```

## 3. 采集Guidance=1 Dense Trajectory

Stage-0进程完全退出后执行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/sample_vllm_trajectories.sh \
  --phase dit \
  --config configs/vllm_block_profile_guidance1.yaml \
  --manifest data/block_profile/guidance1/stage0_conditions/manifest.jsonl \
  --limit 20 --resume
```

检查：

```bash
wc -l data/block_profile/guidance1/trajectories/manifest.jsonl

python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

root = Path("data/block_profile/guidance1/trajectories")
row = json.loads(root.joinpath("manifest.jsonl").read_text().splitlines()[0])
sample = root / row["path"]
metadata = json.loads((sample / "metadata.json").read_text())
with safe_open(str(sample / "trajectory.safetensors"), framework="pt") as handle:
    guidance = handle.get_tensor("guidance")
print("guidance_scale=", metadata["guidance_scale"])
print("guidance_tensor=", guidance)
PY
```

必须显示：

```text
guidance_scale=1.0
guidance_tensor≈1000
```

profile入口还会对全部20条再次强制校验，误用2.5数据会直接停止。

## 4. 16卡运行Block Profile

确保采样服务已退出并释放全部NPU：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACCELERATE_CONFIG="$PWD/configs/accelerate_zero3_16npu.yaml"

bash scripts/profile_sla_blocks.sh configs/block_profile_guidance1.yaml
```

脚本通过ZeRO-3切分80B Dense模型，每个rank处理10个trajectory point。Attention
forward保持Dense，只额外计算pooled Q/K统计。

## 5. 输出

```text
results/block_profiles/dense-guidance1/
├── block_profile.json
└── block_profile.png
```

JSON包含：

```text
global
by_layer[0...31]
by_step[0...7]
recommendation.topk
```

默认候选：

```text
0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5
```

推荐规则为：最小候选比例满足P10 pooled-mass recall达到95%，并且不低于90% query
达到95%质量所需的block比例。如果最大候选仍不满足，应查看JSON而不是直接采用推荐值。

## 6. 结果解释

重点比较：

```text
topk=0.125的P10/P05 recall
topk=0.1875的P10/P05 recall
topk=0.25的P10/P05 recall
最差layer
最差MeanFlow step
```

例如若0.125平均recall较高但P05显著偏低，而0.25的P10/P05稳定达到95%左右，说明
固定12.5%会在少量关键query上丢失大量信息，0.25更适合作为full-QKVO训练候选。

# Badcase T2I 训练验证与实时曲线

本流程把 `datasets/test/badcase_t2i/cases.json` 中固定的 prompt/seed 转成独立的
Dense 8-step MeanFlow trajectory 验证集。训练期间只执行 SLA student forward，目标是
缓存的 Dense `teacher_diffusion_prediction`，不会占用另一套 vLLM 服务或重复运行
Dense teacher。

`baseline_images` 是最终图片的人工对比基线，不适合作为 DiT prediction 的逐元素
MSE 目标。训练内数值验证使用同一 `x_t/t/r/condition` 下的 Dense prediction；最终图片
质量仍需在训练进程停止后用 badcase 脚本评测。

## 1. 固定验证 Prompt

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA

python tools/build_badcase_validation_manifest.py \
  --cases datasets/test/badcase_t2i/cases.json \
  --output datasets/validation/badcase_t2i/prompts.jsonl \
  --limit 20

cat datasets/validation/badcase_t2i/prompts.jsonl
```

manifest 保留每条 case 自己的 seed。不要把验证 prompt 混入正式训练的
`data/trajectories`。

## 2. 采集 Stage-0 condition

Stage-0 使用 vLLM-Omni TP8。20条prompt可以一起提交；采集器每完成一条就原子写入
sample JSON，`--resume` 只跳过已经完成的条目。

```bash
unset HUNYUAN_SLA_ADAPTER
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_OMNI_REPO=/mnt/share/r50063443/vllm-omni

bash scripts/sample_vllm_trajectories.sh \
  --phase stage0 \
  --config configs/vllm_badcase_validation_sampling.yaml \
  --manifest datasets/validation/badcase_t2i/prompts.jsonl \
  --limit 20 \
  --resume

wc -l data/validation/badcase_t2i/stage0_conditions/manifest.jsonl
```

## 3. 采集 Dense 8-step 验证 trajectory

DiT 使用 vLLM-Omni Dense TP8+EP。不同 case 的 seed 可以不同；采集器按 seed 分组，
复用同一个已加载的引擎依次执行，避免为每个 seed 重新加载 80B 模型。每个 prompt
写一个目录，其中包含 8 个 prediction、9 个 latent 和完整 condition。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash scripts/sample_vllm_trajectories.sh \
  --phase dit \
  --config configs/vllm_badcase_validation_sampling.yaml \
  --manifest data/validation/badcase_t2i/stage0_conditions/manifest.jsonl \
  --limit 20 \
  --resume

wc -l data/validation/badcase_t2i/trajectories/manifest.jsonl
find data/validation/badcase_t2i/trajectories/samples -name READY.json | wc -l
```

两个计数都必须为20。20个prompt共160个teacher-forced验证trajectory point。

## 4. 16 NPU 正式训练

`configs/train_sla_trajectory.yaml` 默认配置：

- 训练每卡 batch 4，全局 batch 64。
- Teacher-forced验证每卡batch 1，20 prompt × 8 step = 160 point；16卡下每个rank
  执行10次forward。
- 每25步同时执行20条完整8-step自由rollout；补齐到32个rollout slot以保持ZeRO同步，
  padding结果不进入统计。
- 每步记录JSONL，每5步自动刷新PNG，每25步记录teacher-forced与自由rollout指标。
- 验证输出global/逐step MSE、relative MSE、cosine distance，以及自由rollout最终latent
  relative MSE、cosine distance和Laplacian relative MSE。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export TRAIN_PARALLEL=zero3

bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla \
  --max-steps 250 \
  --output-dir results/training/trajectory-recovery
```

未准备验证 trajectory 时只做训练 smoke，可显式关闭：

```bash
bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 5 --no-validation \
  --output-dir results/training/trajectory-smoke
```

恢复训练：

```bash
bash scripts/train_sla.sh configs/train_sla_trajectory.yaml \
  --stage sla --max-steps 250 \
  --resume-from results/training/trajectory-recovery/sla-step-125 \
  --output-dir results/training/trajectory-recovery
```

恢复时 `metrics.jsonl` 会保留 `step <= 125` 的记录并删除旧运行中更晚的记录，避免曲线
出现重复 step。

## 5. 查看指标和曲线

```text
results/training/trajectory-recovery/metrics/
├── metrics.jsonl
├── training_metrics.png
└── index.html
```

直接查看 PNG，或启动一个轻量静态服务器：

```bash
python -m http.server 6006 \
  --directory results/training/trajectory-recovery/metrics
```

浏览器访问 `http://<训练机>:6006/`。`index.html` 每 15 秒刷新，不需要 TensorBoard。
也可随时从 JSONL 手工重画：

```bash
python tools/plot_training_metrics.py \
  results/training/trajectory-recovery/metrics/metrics.jsonl
```

## 6. checkpoint 图片评测

训练和 16-NPU vLLM 服务不能同时占用同一组设备。导出并启动对应 adapter 后，使用
`--run-name` 隔离 Dense、ZeroInit 和训练 checkpoint 的输出：

```bash
bash scripts/run_badcase_eval.sh \
  --task badcase_t2i --limit 20 --steps 8 \
  --bot-task think --system-prompt-type en_unified \
  --run-name trained-step-250 --overwrite
```

输出位于：

```text
datasets/test/badcase_t2i/runs/trained-step-250/
├── output_images/<index>/seed_<seed>.png
└── run_results.jsonl
```

分别使用 `dense`、`zero-init` 和 `trained-step-250` 作为 run name，即可保留三组结果做
人工或自动图像质量对比。

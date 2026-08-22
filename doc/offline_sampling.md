# 离线采样

离线采样线只负责把外部图文数据转换为 Hunyuan 可训练的 latent cache。它读取图片、运行 Hunyuan tokenizer 与冻结 VAE，写入 `latent_z0` 和静态 condition；训练阶段不再读取图片或执行 VAE encode。

## 原始数据

COYO 是原始数据来源之一，但采样器不绑定数据集名称。COYO 发布的是图片 URL 和元数据，不是预下载图片；先下载一个官方 Parquet metadata shard，筛选候选，再下载可用图片。为获得 2,000 个最终样本，建议先准备 12,000 个候选，为失效 URL、解码失败和 VAE encode 失败留出余量：

```bash
python tools/select_coyo_subset.py \
  --input /datasets/coyo/coyo-part-00000.parquet \
  --output /datasets/hunyuan_sla/candidates.jsonl \
  --candidate-count 12000

python tools/download_images.py \
  --metadata /datasets/hunyuan_sla/candidates.jsonl \
  --output-dir /datasets/hunyuan_sla/raw
```

下载结果会生成如下 JSONL manifest：

```json
{"id":"1","image_path":"1.jpg","caption":"A dog running in a park"}
```

`download_images.py` 会校验图片内容并输出 `/datasets/hunyuan_sla/raw/metadata.jsonl`，可作为采样器的 `source.manifest_path`。

## 采样和验证

编辑 `configs/sampling.yaml` 中的 `model_path`、`source.manifest_path`、`source.image_root`。加载 CANN 环境后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 bash scripts/sample.sh configs/sampling.yaml --resume
python tools/inspect_latent.py --cache-dir data/cache
```

采样会显示 rank 0 的进度条。它只加载 VAE、tokenizer 和 image processor，不加载 Transformer/MoE；按 `sample_id % world_size` 分片写入 `data/cache/rank-XXX/shards/`。全部 rank 到达 barrier 后，rank 0 合并为训练使用的 `data/cache/shards/` 与 `manifest.jsonl`，并自动验证。中断后再次带 `--resume` 执行，各 rank 会读取自身 manifest 跳过已完成 sample。

只有 `data/cache/READY.json` 存在时，训练才会接受该 cache。

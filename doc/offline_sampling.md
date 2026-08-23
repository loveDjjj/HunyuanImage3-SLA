# 离线采样

离线采样线只负责把外部图文数据转换为 Hunyuan 可训练的 latent cache。它读取图片、运行 Hunyuan tokenizer 与冻结 VAE，写入 `latent_z0` 和静态 condition；训练阶段不再读取图片或执行 VAE encode。

## 原始数据

当前默认数据源是 Flickr30k。下载 `cjc/flickr30k` 的本地图像 archive 后，结合 Karpathy `dataset_flickr30k.json` 生成统一 manifest。工具只从 `train` split 的不同图片中选择 2,000 条，且每张图片确定性选取一条 caption：

```bash
tar -xf datasets/flickr30k/flickr30k-images.tar -C datasets/flickr30k

python tools/prepare_flickr30k_manifest.py \
  --annotations datasets/flickr30k/dataset_flickr30k.json \
  --images-dir datasets/flickr30k/flickr30k-images \
  --output datasets/flickr30k/metadata.jsonl \
  --split train --sample-count 2000
```

下载结果会生成如下 JSONL manifest：

```json
{"id":"1","image_path":"1.jpg","caption":"A dog running in a park"}
```

`metadata.jsonl` 可直接作为采样器的 `source.manifest_path`。Flickr30k 图片源自 Flickr，使用必须遵守 Flickr 条款及研究/教育用途限制。

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

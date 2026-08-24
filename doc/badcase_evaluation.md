# Badcase 批量测评

本流程支持 `badcase_ti2i` 和 `badcase_t2i`，使用 cases 中原始 prompt 与 seed 调用
vLLM-Omni。HunyuanImage-3.0-Instruct-Distil 默认执行 8 步去噪。

## 目录布局

```text
datasets/test/
├── badcase_ti2i/
│   ├── cases.json
│   ├── input_images/<index>/*
│   ├── baseline_images/<index>/*
│   ├── output_images/<index>/seed_<seed>.png
│   └── run_results.jsonl
└── badcase_t2i/
    ├── cases.json
    ├── baseline_images/<index>/*
    ├── output_images/<index>/seed_<seed>.png
    └── run_results.jsonl
```

准备脚本保留 `image_urls` 和 `baseline_url`，将 i2i 的 `inputs` 改成已下载图片的
绝对路径，并新增 `baseline_image` 本地绝对路径。下载和 JSON 更新支持重复运行。

## 准备数据

先将原始 cases 放入目标目录：

```bash
ROOT=/mnt/share/r50063443/HunyuanImage3-SLA
mkdir -p "$ROOT/datasets/test/badcase_ti2i" "$ROOT/datasets/test/badcase_t2i"

cp /mnt/share/r50063443/0814-badcases/extracted/badcases/badcases/badcase_ti2i/cases.json \
  "$ROOT/datasets/test/badcase_ti2i/cases.json"
cp /mnt/share/r50063443/0814-badcases/extracted/badcases/badcases/badcase_t2i/cases.json \
  "$ROOT/datasets/test/badcase_t2i/cases.json"

cd "$ROOT"
bash scripts/prepare_badcase_eval.sh
```

任何下载失败都会打印 case index 并返回非零退出码；已成功的文件和路径仍会保存，修复网络后
直接重跑即可。只准备一个任务可使用 `--task badcase_ti2i` 或 `--task badcase_t2i`。

## 调用 vLLM-Omni

先用每种任务各一条做冒烟测试：

```bash
export VLLM_OMNI_URL=http://127.0.0.1:8000

bash scripts/run_badcase_eval.sh --task badcase_ti2i --limit 1
bash scripts/run_badcase_eval.sh --task badcase_t2i --limit 1
```

确认服务日志和输出图片正常后执行全量：

```bash
bash scripts/run_badcase_eval.sh
```

脚本按任务分别调用：

- t2i：`POST /v1/images/generations`，JSON 请求。
- i2i：`POST /v1/images/edits`，multipart 上传本地输入图片。

默认参数为 `--steps 8 --t2i-size 1024x1024 --i2i-size auto`。已有且可正常解码的
输出会跳过；使用 `--overwrite` 强制重跑。可用 `--offset`、`--limit` 分段执行，失败详情
记录在各任务目录的 `run_results.jsonl`。

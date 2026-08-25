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

## 使用 QKV/O SLA adapter 启动 vLLM-Omni

训练结束后的 DeepSpeed `sla-step-*` 目录不能直接交给 vLLM-Omni。服务读取的是已经
导出的 v2 adapter 目录，其中必须同时存在：

```text
adapter.safetensors
adapter_config.json
SHA256SUMS
```

以下命令假设服务器上的路径为：

```bash
SLA_ROOT=/mnt/share/r50063443/HunyuanImage3-SLA
VLLM_ROOT=/mnt/share/r50063443/vllm-omni
MODEL_ROOT=/mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil
ADAPTER_ROOT="$SLA_ROOT/results/adapters/qkvo-delta-step-200"
```

将 `qkvo-delta-step-200` 替换为实际导出的目录名。`ADAPTER_ROOT` 必须指向目录，
不能指向其中的 `adapter.safetensors` 文件。

### 1. 校验导出产物

在训练环境执行本仓库的完整校验：

```bash
cd "$SLA_ROOT"

python tools/inspect_sla_adapter.py --adapter-dir "$ADAPTER_ROOT"
(cd "$ADAPTER_ROOT" && sha256sum -c SHA256SUMS)

python - "$ADAPTER_ROOT/adapter_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)

assert config["format_version"] == 2, config
assert config["trained_components"] == ["proj_l", "qkv_delta", "o_delta"], config
assert config["num_layers"] == 32, config
assert config["head_dim"] == 128, config
assert config["topk"] == 0.125, config
assert config["blkq"] == 128, config
assert config["blkk"] == 128, config
print("QKV/O SLA adapter configuration is valid")
PY
```

正常的 QKV/O adapter 应显示 `tensor_count=128`、`parameter_count=1342705664`，
dtype 同时包含 BF16 和 FP32。若 `blkq=64` 或只有 64 个 tensor，这是旧的
proj-only artifact，不能使用下面的 QKVO deploy profile。

### 2. 更新并安装 vLLM-Omni 适配分支

当前适配基于 vLLM 0.26 容器，必须使用专用分支：

```bash
cd "$VLLM_ROOT"
git fetch origin
git switch hunyuan-image3-distil-sla-v0.26
git pull --ff-only origin hunyuan-image3-distil-sla-v0.26

python -m pip install -e . --no-deps
```

确认新的 deploy profile 和 adapter loader 都存在：

```bash
test -f vllm_omni/deploy/hunyuan_image_3_distil_sla_qkvo.yaml
grep -n 'apply_attention_deltas' \
  vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py
grep -A8 'hunyuan.diffusion:' \
  vllm_omni/deploy/hunyuan_image_3_distil_sla_qkvo.yaml
```

QKVO profile 必须是 `topk=0.125`、`blkq=128`、`blkk=128`。不要使用旧的
`hunyuan_image_3_distil_sla.yaml`，它保留 `blkq=64`，只兼容旧 proj-only adapter。

### 3. 启动 16 NPU 服务

先确保训练进程已经退出并释放全部 NPU，然后在 vLLM-Omni 容器中执行：

```bash
cd "$VLLM_ROOT"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HUNYUAN_SLA_ADAPTER="$ADAPTER_ROOT"

test -d "$MODEL_ROOT"
test -f "$HUNYUAN_SLA_ADAPTER/adapter.safetensors"
test -f "$HUNYUAN_SLA_ADAPTER/adapter_config.json"

mkdir -p logs
vllm serve "$MODEL_ROOT" \
  --omni \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_sla_qkvo.yaml \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  2>&1 | tee logs/hunyuan-image3-qkvo-sla.log
```

该 profile 使用两阶段共 16 张 NPU：

- Stage 0：设备 `0-7`，Hunyuan AR/comprehension，TP8。
- Stage 1：设备 `8-15`，diffusion，TP8 + expert parallel + MindIE SLA。
- Instruct-Distil 默认 `num_inference_steps=8`、`guidance_scale=2.5`。
- v2 QKV/O delta 在 vLLM 执行 TP 分片前逐层加到基础 QKV/O 权重；不需要先合并
  或复制完整 80B checkpoint。

服务日志必须出现 adapter 校验成功以及 `MINDIE_SLA` backend。以下情况应立即停止：

- `weights were not initialized from checkpoint`：adapter 没有加载或目录错误。
- runtime 与 adapter 的 `topk/blkq/blkk` 不一致：使用了错误 deploy profile。
- `format_version`、SHA256、shape 或 `trained_components` 校验失败：导出产物错误。

### 4. 健康检查和单条冒烟测试

另开终端：

```bash
export VLLM_OMNI_URL=http://127.0.0.1:8000

curl --fail --silent --show-error "$VLLM_OMNI_URL/health"
```

然后分别测试一条 T2I 和 I2I。Hunyuan 专用 prompt 参数必须显式传入，避免 I2I
请求被通用处理器标成不受支持的 `img2img` modality：

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA

bash scripts/run_badcase_eval.sh \
  --task badcase_t2i \
  --limit 1 \
  --steps 8 \
  --bot-task think \
  --system-prompt-type en_unified \
  --overwrite

bash scripts/run_badcase_eval.sh \
  --task badcase_ti2i \
  --limit 1 \
  --steps 8 \
  --bot-task think \
  --system-prompt-type en_unified \
  --overwrite
```

确认两个任务均返回 `completed: 1, failed: 0`，并能正常打开对应
`output_images/<index>/seed_<seed>.png` 后，再执行全量测评。

## 调用 vLLM-Omni

先用每种任务各一条做冒烟测试：

```bash
export VLLM_OMNI_URL=http://127.0.0.1:8000

bash scripts/run_badcase_eval.sh --task badcase_ti2i --limit 1 \
  --bot-task think --system-prompt-type en_unified
bash scripts/run_badcase_eval.sh --task badcase_t2i --limit 1 \
  --bot-task think --system-prompt-type en_unified
```

确认服务日志和输出图片正常后执行全量：

```bash
bash scripts/run_badcase_eval.sh \
  --bot-task think \
  --system-prompt-type en_unified
```

脚本按任务分别调用：

- t2i：`POST /v1/images/generations`，JSON 请求。
- i2i：`POST /v1/images/edits`，multipart 上传本地输入图片。

默认参数为 `--steps 8 --t2i-size 1024x1024 --i2i-size auto`。已有且可正常解码的
输出会跳过；使用 `--overwrite` 强制重跑。可用 `--offset`、`--limit` 分段执行，失败详情
记录在各任务目录的 `run_results.jsonl`。

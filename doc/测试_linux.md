# HunyuanImage3-SLA Linux 部署与测试指南

本文档说明 Ascend NPU Linux 服务器上的最小部署环境，以及 Dense Attention、SparseLinearAttention（SLA）恢复训练的验收流程。本项目不引入 FastVideo、MindSpeed-MM 或 vLLM-Omni。

## 1. 下载项目与三个上游仓库

在服务器上克隆项目后，进入项目根目录并下载三个上游仓库：

```bash
git clone https://github.com/loveDjjj/HunyuanImage3-SLA.git
cd HunyuanImage3-SLA
mkdir -p upstream

# DiffSynth 训练框架
git clone --depth 1 https://github.com/modelscope/DiffSynth-Studio.git upstream/DiffSynth-Studio

# MindIE-SD，提供已验证的 SparseLinearAttention NPU forward/backward
git clone --depth 1 https://gitcode.com/Ascend/MindIE-SD.git upstream/MindIE-SD

# HunyuanImage-3.0。使用 partial clone，只检出模型代码和配置，避免下载仓库中的大型资源。
git clone --filter=blob:none --no-checkout https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git upstream/HunyuanImage-3.0
git -C upstream/HunyuanImage-3.0 sparse-checkout set --no-cone 'hunyuan_image_3/**' '*.py' '*.md' 'requirements.txt' 'setup.py' 'configs/**'
git -C upstream/HunyuanImage-3.0 checkout main
```

确认三个目录都存在：

```bash
git -C upstream/DiffSynth-Studio rev-parse --short HEAD
git -C upstream/MindIE-SD rev-parse --short HEAD
git -C upstream/HunyuanImage-3.0 rev-parse --short HEAD
```

模型权重不在上述代码仓库下载流程中。请将 `HunyuanImage-3.0-Instruct-Distil` 权重单独下载到服务器本地目录，并在 `configs/train_sla.yaml` 配置其路径。

## 2. 环境要求

| 组件 | 最小要求 |
| --- | --- |
| 操作系统 | Linux |
| Python | 3.10 或更高版本 |
| Ascend CANN | 9.0.0，或与所选 `torch_npu` wheel 匹配的版本 |
| PyTorch / torch_npu | 版本必须匹配，且受已安装 CANN 支持 |
| MindIE-SD SLA | `triton==3.5.0` 与 `triton-ascend==3.2.1` |
| Hunyuan 依赖 | `transformers==4.57.1` 与 `upstream/HunyuanImage-3.0/requirements.txt` |

`triton-ascend==3.2.1` 不在 PyPI 发布，需要从 Ascend/GitCode Release 下载与服务器 Python 版本、CPU 架构匹配的 wheel。

## 3. 激活 CANN 与确认 NPU

新建 shell 后，激活 Python 环境并加载 CANN 环境变量。若 CANN 安装位置不同，请修改路径。

```bash
cd /path/to/HunyuanImage3-SLA
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
npu-smi info
```

以下命令必须输出 `npu_available: True`：

```bash
python - <<'PY'
import torch
import torch_npu

print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('npu_available:', torch.npu.is_available())
print('device_count:', torch.npu.device_count())
if not torch.npu.is_available():
    raise SystemExit('torch_npu 无法访问 NPU')
print('device:', torch.npu.get_device_name(0))
PY
```

## 4. 安装 Python 依赖

先从 Ascend 官方软件源获取一对匹配的 `torch` 和 `torch_npu` wheel。不要将 CUDA PyTorch wheel 与 `torch_npu` 混用。

```bash
python -m pip install --upgrade pip setuptools wheel

# 替换为与 CANN 匹配的 Ascend PyTorch 与 torch_npu wheel 路径。
python -m pip install /wheels/torch-*.whl /wheels/torch_npu-*.whl

# MindIE-SD SLA 运行时。先安装 Triton-Ascend，再安装本地 MindIE-SD。
python -m pip install triton==3.5.0
python -m pip install /wheels/triton_ascend-3.2.1-*.whl
python -m pip install -e upstream/MindIE-SD

# Hunyuan、DiffSynth 与训练脚本依赖。
python -m pip install -r upstream/HunyuanImage-3.0/requirements.txt
python -m pip install imageio accelerate peft pyyaml tqdm pytest
python -m pip install -e upstream/DiffSynth-Studio
python -m pip install -e upstream/HunyuanImage-3.0
```

检查三个上游项目是否可导入：

```bash
export PYTHONPATH="$PWD/train:$PWD/upstream/DiffSynth-Studio:$PWD/upstream/MindIE-SD:$PWD/upstream/HunyuanImage-3.0:${PYTHONPATH:-}"
python - <<'PY'
import torch
import transformers
from diffsynth.diffusion import DiffusionTrainingModule
from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3SDPAAttention
from mindiesd.layers import SparseLinearAttention

assert torch.npu.is_available()
assert transformers.__version__ == '4.57.1', transformers.__version__
print('DiffSynth:', DiffusionTrainingModule.__name__)
print('Hunyuan attention:', HunyuanImage3SDPAAttention.__name__)
print('MindIE SLA:', SparseLinearAttention.__name__)
PY
```

## 5. 验证 MindIE-SD SLA Kernel

本项目只复用 MindIE-SD 的 SLA kernel，不重新编译或实现 kernel。在接入训练前，先运行上游 NPU 测试：

```bash
export PYTHONPATH="$PWD/upstream/MindIE-SD:${PYTHONPATH:-}"
python -m pytest -q upstream/MindIE-SD/tests/layers/flash_attn/test_sparse_linear_attn.py
```

快速诊断只运行 NPU 用例：

```bash
python -m pytest -q upstream/MindIE-SD/tests/layers/flash_attn/test_sparse_linear_attn.py -k NPU
```

测试必须完成 SLA forward 和 backward，且不得出现不支持的后端、设备或 shape 错误。默认配置使用 `head_dim=128`、`BLKQ=64`、`BLKK=128`，符合 MindIE-SD AscendC 路径的约束。

## 6. 配置模型与离线 latent cache

将 `configs/train_sla.yaml` 中的 `model_path` 修改为本地 `HunyuanImage-3.0-Instruct-Distil` 权重目录。该目录应包含模型配置、tokenizer、custom code 配置和所有权重分片。

训练不再读取原图或旧的 `.pt` forward 参数。先通过离线采样生成 `data/cache`，其中包含 `latent_z0`、静态 Hunyuan condition tensors、`manifest.jsonl` 和验证标记 `READY.json`。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/sample.sh configs/sampling.yaml --resume
bash scripts/verify_cache.sh data/cache
```

采样和训练的完整数据契约见 [离线采样指南](离线采样.md)。训练脚本会拒绝没有 `READY.json` 的 cache。可复制执行的完整验收命令见 [服务器测试命令](服务器测试命令.md)。

## 7. 第一阶段：Dense Attention 单步测试

该阶段不替换 attention。模型被冻结，仅解冻与 `dense_trainable_patterns` 匹配的参数（默认 `final_layer`），用于确认 Hunyuan diffusion forward、loss、backward、optimizer step 和 checkpoint 保存都可工作。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_sla.sh configs/train_sla.yaml --stage dense --max-steps 1
```

预期关键输出：

```text
stage=dense trainable_parameters=...
step=1 loss=... finite_grad=True
checkpoint=results/training/default/dense-step-1.pt
```

如果 loss 不是有限值、`finite_grad` 不为 `True`，或没有 checkpoint，则应停止并排查环境。

## 8. 第二阶段：SLA Recovery 单步测试

`train/sla_adapter.py` 会在运行时替换 Hunyuan attention。Dense teacher 在 `torch.no_grad()` 下临时恢复原 attention；SLA student 生成模型级 `diffusion_prediction`，两者 MSE 为 recovery loss。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

预期关键输出：

```text
stage=sla trainable_parameters=...
...sla.proj_l.weight
...sla.proj_l.bias
step=1 loss=... finite_grad=True
checkpoint=results/training/default/sla-step-1.pt
```

optimizer 中仅包含每个被替换 attention 的 `sla.proj_l.weight` 和 `sla.proj_l.bias`。AR/reasoning、recaption、VAE、MoE、projection、norm 和其他 Transformer 参数均被冻结。

## 9. 多 NPU DDP 启动路径（未实机验证）

必须先通过单 NPU SLA one-step。以下仅验证训练脚本的 DDP 启动路径，前提是**每张** NPU 都能容纳完整 BF16 Hunyuan checkpoint、Dense teacher 前向和 SLA student 反向；DDP 不会切分模型权重，不能解决 80B 模型显存问题：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

只有 rank 0 会写入 `sla-step-1.pt`。当前未完成多 NPU 实机验收，不应将此命令视为已验证的正式训练方案。SP、TP、EP 和 ZeRO 尚未接入；详细边界见 [服务器测试命令](服务器测试命令.md)。

## 10. 常见问题

| 现象 | 排查方式 |
| --- | --- |
| `torch.npu.is_available()` 为 false | 重新 `source` CANN 的 `set_env.sh`，确认 `torch_npu` 与 CANN 匹配，并执行 `npu-smi info`。 |
| `SparseLinearAttention is disabled` | 安装匹配的 Triton-Ascend 3.2.1 wheel，重启 Python 环境后重新检查。 |
| SLA 提示输入不在 NPU | 检查 `ASCEND_RT_VISIBLE_DEVICES`、`torch.npu.is_available()`，以及 Accelerate 没有被配置为 CPU。 |
| SLA 提示不支持 shape 或 backend | 使用受支持的 head dim 和 block。默认 `head_dim=128`、`BLKQ=64`、`BLKK=128` 是支持的组合。 |
| Hunyuan 导入时找不到 `transformers.models.siglip2` | 重装 Hunyuan 依赖，尤其确认 `transformers==4.57.1`。 |
| SLA 拒绝 `attention_mask` | 当前 SLA 接口不支持任意 attention mask，recovery 输入不能传该字段。 |
| 没有有限梯度 | 确认 `mode='gen_image'`、`diffusion_prediction` 非空，并检查日志中列出了 `sla.proj_l`。 |

# HunyuanImage3-SLA

HunyuanImage3-SLA 是一个面向 Ascend NPU 的 HunyuanImage-3.0-Instruct-Distil SLA 恢复训练项目。

项目以 DiffSynth-Studio 的 `DiffusionTrainingModule` 为训练模块基础，在不重写 HunyuanImage3 模型、也不重新实现 MindIE-SD kernel 的前提下，将 Hunyuan diffusion Transformer 中的 Dense GQA attention 运行时替换为 MindIE-SD `SparseLinearAttention`。

## 目标

- 复用 HunyuanImage-3.0-Instruct-Distil。
- 在 Ascend NPU 上执行 SLA forward、backward 和 fine-tuning。
- 保持 Hunyuan GQA 结构：32 个 Q heads、8 个 KV heads。
- 复用 Hunyuan 原有 `repeat_kv`，将 K/V 扩展为 32 heads 后输入 SLA。
- 使用 Dense Hunyuan `diffusion_prediction` 作为 teacher，SLA Hunyuan 输出作为 student，以 MSE 做 recovery training。
- 第一版仅训练每层 SLA 的 `proj_l.weight` 和 `proj_l.bias`。

## 目录

```text
train/
  train_sla.py        # Dense/SLA 单步与多卡训练入口
  hunyuan_adapter.py  # Hunyuan 加载、冻结、teacher/student MSE
  sla_adapter.py      # Dense attention 到 MindIE-SD SLA 的运行时替换
sampling/             # 离线 Hunyuan VAE latent 采样线
tools/                # COYO 筛选、下载与 cache 检查等独立工具
configs/
  sampling.yaml       # 离线采样配置
  train_sla.yaml      # 模型、cache 和 SLA 参数
scripts/
  sample.sh           # 离线采样启动脚本
  train_sla.sh        # 单 NPU / torchrun 训练脚本
doc/
  测试_linux.md        # Linux 环境部署、上游下载与验收流程
```

`upstream/` 是本地依赖目录，不提交到 Git。三个上游仓库的下载命令和完整环境准备流程见 [doc/测试_linux.md](doc/测试_linux.md)。

## Attention 替换位置

Hunyuan 原始 attention 位于：

```text
upstream/HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
HunyuanImage3SDPAAttention.forward
```

`train/sla_adapter.py` 会识别该模块并替换为 `HunyuanImage3SLAAttention`。适配器保留原始 QKV projection、RoPE、QK norm 和 `repeat_kv`，只将原有 SDPA 调用换成 MindIE-SD `SparseLinearAttention`。

## 快速启动

先按照 [Linux 部署与测试指南](doc/测试_linux.md) 下载上游仓库、安装 CANN/PyTorch NPU/MindIE-SD 依赖。离线采样和训练是两条独立流程，详细命令见 [离线采样](doc/离线采样.md)、[训练](doc/训练.md) 与可直接执行的 [服务器测试命令](doc/服务器测试命令.md)。

修改 `configs/sampling.yaml`：

```yaml
model_path: /path/to/HunyuanImage-3.0-Instruct-Distil
source:
  manifest_path: /path/to/metadata.jsonl
  image_root: /path/to/images
```

生成训练 cache：

```bash
bash scripts/sample.sh configs/sampling.yaml --resume
bash scripts/verify_cache.sh data/cache
```

训练读取 `data/cache` 中的 `latent_z0` 和静态 Hunyuan condition，不读取图片，也不执行 VAE encode。

### Dense Attention 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_sla.sh configs/train_sla.yaml --stage dense --max-steps 1
```

### SLA Recovery 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

成功时日志会输出 `loss=... finite_grad=True`，并在 `results/training/default/` 写入仅包含可训练参数的 checkpoint。

### 多 NPU 启动

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 bash scripts/train_sla.sh configs/train_sla.yaml --stage sla --max-steps 1
```

## 当前训练参数

SLA 阶段冻结模型中的原始 Hunyuan 参数，只训练如下参数：

```text
...self_attn.sla.proj_l.weight
...self_attn.sla.proj_l.bias
```

默认 SLA 配置：`topk=0.125`、`BLKQ=64`、`BLKK=128`、BF16。详细环境检查与错误排查见 [doc/测试_linux.md](doc/测试_linux.md)。

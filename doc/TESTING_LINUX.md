# HunyuanImage3-SLA Linux Deployment and Test Guide

This document describes the minimum Linux/Ascend environment and the validation sequence for Dense and SparseLinearAttention (SLA) recovery training. It does not install FastVideo, MindSpeed-MM, or vLLM-Omni.

## 1. Preconditions

Use an Ascend Linux host with an available NPU. The commands assume the project root is the current directory.

| Component | Required baseline |
| --- | --- |
| OS | Linux |
| Python | 3.10 or newer |
| CANN | 9.0.0, or the version required by the selected `torch_npu` wheel |
| PyTorch / torch_npu | Matching releases supported by the installed CANN |
| MindIE-SD SLA | `triton==3.5.0` and `triton-ascend==3.2.1` |
| Hunyuan code dependencies | `transformers==4.57.1` plus `upstream/HunyuanImage-3.0/requirements.txt` |

`triton-ascend==3.2.1` must be installed from the Ascend/GitCode release wheel matching the host Python version and architecture. It is not available from PyPI.

## 2. Activate CANN and Check Devices

Start a new shell, activate the virtual environment, then load the CANN variables. Adjust the CANN installation path when necessary.

```bash
cd /path/to/HunyuanImage3-SLA
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
npu-smi info
```

The following check must report `npu_available=True` and list the expected device.

```bash
python - <<'PY'
import torch
import torch_npu

print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('npu_available:', torch.npu.is_available())
print('device_count:', torch.npu.device_count())
if not torch.npu.is_available():
    raise SystemExit('torch_npu cannot access an NPU')
print('device:', torch.npu.get_device_name(0))
PY
```

## 3. Install Python Dependencies

Install a matching `torch` and `torch_npu` pair from the official Ascend package source for the chosen CANN release. Do not mix a CUDA PyTorch wheel with `torch_npu`.

```bash
python -m pip install --upgrade pip setuptools wheel

# Replace both wheel paths with a matched Ascend torch / torch_npu pair.
python -m pip install /wheels/torch-*.whl /wheels/torch_npu-*.whl

# MindIE-SD SLA runtime. Install the Triton-Ascend wheel before the local package.
python -m pip install triton==3.5.0
python -m pip install /wheels/triton_ascend-3.2.1-*.whl
python -m pip install -e upstream/MindIE-SD

# Hunyuan and DiffSynth runtime dependencies.
python -m pip install -r upstream/HunyuanImage-3.0/requirements.txt
python -m pip install imageio accelerate peft pyyaml tqdm pytest
python -m pip install -e upstream/DiffSynth-Studio
python -m pip install -e upstream/HunyuanImage-3.0
```

Check that the exact Hunyuan Transformers version and all three project imports work:

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

## 4. Verify the MindIE-SD SLA Kernel

The project reuses MindIE-SD's kernel and does not build or replace it. Run the upstream NPU test before integration:

```bash
export PYTHONPATH="$PWD/upstream/MindIE-SD:${PYTHONPATH:-}"
python -m pytest -q upstream/MindIE-SD/tests/layers/flash_attn/test_sparse_linear_attn.py
```

For a quicker diagnostic, execute only the NPU test class:

```bash
python -m pytest -q upstream/MindIE-SD/tests/layers/flash_attn/test_sparse_linear_attn.py -k NPU
```

The test must complete forward and backward without an unsupported backend, device, or shape error. For the default project configuration, SLA receives head dimension 128 and blocks `BLKQ=64`, `BLKK=128`, which meet the MindIE-SD AscendC constraints.

## 5. Configure the Model and Recovery Batches

Set `model_path` in `configs/train_sla.yaml` to a local `HunyuanImage-3.0-Instruct-Distil` checkpoint. The checkpoint must include the model config, custom code configuration, tokenizer assets, and all model weight shards.

`data.serialized_inputs_glob` points to files created with `torch.save(model_forward_kwargs, path)`. Each file must contain one dictionary accepted by `HunyuanImage3ForCausalMM.forward`, for example:

```python
{
    'input_ids': ...,              # Tensor on CPU when saved
    'images': ...,                 # Diffusion latent/image input expected by the upstream model
    'image_mask': ...,
    'timesteps': ...,
    'timesteps_index': ...,
    'mode': 'gen_image',
    'first_step': True,
    'return_dict': True,
    # Do not include attention_mask during SLA recovery.
}
```

The training process moves tensors to the selected NPU. The input must produce a non-empty `diffusion_prediction`. Create the serialized batches through the upstream tokenizer/image preprocessing path; do not use arbitrary token IDs or image tensors.

Validate the batch files before starting training:

```bash
python - <<'PY'
import glob
import torch

paths = sorted(glob.glob('data/recovery_inputs/*.pt'))
assert paths, 'no recovery input batches found'
batch = torch.load(paths[0], map_location='cpu', weights_only=False)
assert isinstance(batch, dict)
assert batch.get('mode') == 'gen_image'
assert batch.get('attention_mask') is None
print('batch:', paths[0])
print('keys:', sorted(batch))
PY
```

## 6. Phase 1: Dense Forward and Backward

This phase does not replace attention. It freezes the model except for parameters matching `dense_trainable_patterns` (default: `final_layer`), and confirms an actual Hunyuan diffusion forward, loss, backward, optimizer step, and checkpoint save.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train.sh configs/train_sla.yaml --stage dense --max-steps 1
```

Expected terminal signals:

```text
stage=dense trainable_parameters=...
step=1 loss=... finite_grad=True
checkpoint=outputs/hunyuan-image3-sla/dense-step-1.pt
```

Stop and diagnose the environment if the loss is not finite, `finite_grad` is not `True`, or the checkpoint is absent.

## 7. Phase 2: SLA Recovery One-Step Test

SLA replacement happens at runtime in `train/sla_adapter.py`. The Dense teacher is temporarily restored under `torch.no_grad()` and the SLA student produces the model-level diffusion prediction. Their MSE is the recovery loss.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash scripts/train.sh configs/train_sla.yaml --stage sla --max-steps 1
```

Expected terminal signals:

```text
stage=sla trainable_parameters=...
...sla.proj_l.weight
...sla.proj_l.bias
step=1 loss=... finite_grad=True
checkpoint=outputs/hunyuan-image3-sla/sla-step-1.pt
```

Only `sla.proj_l.weight` and `sla.proj_l.bias` from each replaced Hunyuan attention layer are in the optimizer. AR/reasoning, recaption, VAE, MoE, projections, norms, and the remaining Transformer parameters remain frozen.

## 8. Multi-NPU Test

First complete the one-NPU SLA test. Then run one recovery step with one process per visible NPU:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8 bash scripts/train.sh configs/train_sla.yaml --stage sla --max-steps 1
```

Only rank zero writes `sla-step-1.pt`. Check all ranks for an NPU backend error and verify rank zero prints a finite loss and `finite_grad=True`.

## 9. Failure Checklist

| Symptom | Check |
| --- | --- |
| `torch.npu.is_available()` is false | Source CANN `set_env.sh`; install a matching `torch_npu`; inspect `npu-smi info`. |
| `SparseLinearAttention is disabled` | Install the matching Triton-Ascend 3.2.1 wheel, then reinstall/restart the Python environment. |
| SLA says tensors are not on NPU | Check `ASCEND_RT_VISIBLE_DEVICES`, `torch.npu.is_available()`, and do not launch with CPU Accelerate settings. |
| Unsupported SLA shape/backend | Use a supported head dimension and blocks. Default Hunyuan head dim 128 with `BLKQ=64`, `BLKK=128` is supported. |
| Hunyuan import cannot find `transformers.models.siglip2` | Reinstall the exact Hunyuan requirement set, especially `transformers==4.57.1`. |
| SLA rejects `attention_mask` | Recovery input batches must omit `attention_mask`; arbitrary masked attention is not supported by the current MindIE-SD SLA interface. |
| No finite gradient | Confirm `mode='gen_image'`, that `diffusion_prediction` is non-empty, and that output lists `sla.proj_l` parameters as trainable. |

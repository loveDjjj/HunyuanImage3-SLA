# ZeroInit SLA Baseline

该baseline保持HunyuanImage-3.0-Instruct-Distil原始QKV/O权重不变，将32层SLA
`proj_l.weight/bias`全部设为FP32零。运行时仍完整执行sparse branch、linear feature map、
zero projection和hybrid mask，用于测量未训练SLA引入的原始质量损失。

## 生成adapter

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA

bash scripts/create_zero_sla_adapter.sh \
  results/adapters/sla-zero-init
```

检查结果应包含：

```text
baseline_type=sla_zero_init
training_step=0
tensor_count=64
parameter_count=528384
dtype=[torch.float32]
```

产物约2.02MiB。它不包含`qkv_delta/o_delta`，因此vLLM加载基础模型时不会修改原始
QKV/O projection。

## 启动ZeroInit服务

```bash
cd /mnt/share/r50063443/vllm-omni
export MODEL_ROOT=/mnt/share/r50063443/HunyuanImage-3.0-Instruct-Distil
export HUNYUAN_SLA_ADAPTER=/mnt/share/r50063443/HunyuanImage3-SLA/results/adapters/sla-zero-init
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

vllm serve "$MODEL_ROOT" \
  --omni \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_distil_sla_zero.yaml \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
```

## 评测

使用与Dense和训练SLA完全相同的prompt、seed、8 steps、guidance和输出分辨率：

```bash
cd /mnt/share/r50063443/HunyuanImage3-SLA
export VLLM_OMNI_URL=http://127.0.0.1:8000

bash scripts/run_badcase_eval.sh \
  --task badcase_t2i \
  --limit 1 \
  --steps 8 \
  --bot-task think \
  --system-prompt-type en_unified \
  --overwrite
```

至少比较三组：Dense Base、ZeroInit SLA、训练后的QKVO SLA。ZeroInit到训练SLA的质量
改善代表recovery训练效果；Dense到ZeroInit的差距代表未补偿稀疏化损失。

# TWLA

## Dependencies

```bash
# Clone the github repo and go to the default directory 'TWLA'.
conda create -n twla python=3.9
conda activate twla
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Example: PTQ for Qwen3-8B

### KOTMS
```bash
    python scripts/KOTMS.py \
    --model /data01/home/zhixiong.zhao/models/models/datasets/Qwen3-8B \
    --export_rotated checkpoints/qwen3_8b_rotated.pt \
    --ngpus 4 \
    --use_gmm \
    --gmm_iters 100 \
    --gmm_lr_r 1e-2 \
    --gmm_lr_l 1e-2
```
### ILA-AMP
```bash
    python scripts/ILA_AMP.py \
    --model /data01/home/zhixiong.zhao/models/models/datasets/Qwen3-8B \
    --import_rotated checkpoints/qwen3_8b_rotated.pt \
    --dp_cache dp_cache/qwen3_8b \
    --dp_ngpus 4
```
### Run-TWLA
#### Weight-only (W1.58A16)
```bash
    python run_twla.py \
    --model /data01/home/zhixiong.zhao/models/models/datasets/Qwen3-8B \
    --import_rotated checkpoints/qwen3_8b_rotated.pt \
    --dp_cache dp_cache/qwen3_8b \
    --save_quant_model /data01/home/zhixiong.zhao/TWLA/checkpoints \
    --eval_qa \
    --abits 16
```
#### Weight-Activateion (W1.58A4)
```bash
    python run_twla.py \
    --model /data01/home/zhixiong.zhao/models/models/datasets/Qwen3-8B \
    --import_rotated checkpoints/qwen3_8b_rotated.pt \
    --dp_cache dp_cache/qwen3_8b \
    --load_quant_model /data01/home/zhixiong.zhao/TWLA/checkpoints.tmp \
    --eval_qa \
    --dp_avg_abits 4
```

## Evaluation on zero-shot QA datasets

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) kit to evaluate performance on QA datasets. Please refer to their framework for evaluating quantized models.

## 💡 Acknowledgements

This work is released under the Apache 2.0 license.
The codes are based on [ARB-LLM](https://github.com/ZHITENGLI/ARB-LLM). Please also follow their licenses. Thanks for their awesome works.

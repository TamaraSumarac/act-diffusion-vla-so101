#!/bin/bash
# Diffusion Policy training launch — same frozen dataset + trim as ACT baseline.
# Run on the training box from the directory containing train_policy.py + trim_frames.json.
set -e

DATASET_FREEZE_HASH=e2dd884ad6676140762e8172e50978c2424bbcc2

python train_policy.py \
  --dataset.repo_id=TamaraSumarac/so101_policy_robustness \
  --dataset.revision=$DATASET_FREEZE_HASH \
  --policy.type=diffusion \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/diffusion_baseline \
  --job_name=diffusion_baseline_so101 \
  --batch_size=8 \
  --save_freq=10000 \
  --wandb.enable=true

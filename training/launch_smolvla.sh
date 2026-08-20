# SmolVLA fine-tuning launch — same frozen dataset + trim as ACT/Diffusion.
# Starts from lerobot/smolvla_base (VLM + robot-pretrained action expert);
# preset freezes the vision encoder and trains the action expert only.
# Task string must match the eval instruction verbatim.
set -e

DATASET_FREEZE_HASH=e2dd884ad6676140762e8172e50978c2424bbcc2

python train_policy.py \
  --dataset.repo_id=TamaraSumarac/so101_policy_robustness \
  --dataset.revision=$DATASET_FREEZE_HASH \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --output_dir=outputs/smolvla_baseline \
  --job_name=smolvla_baseline_so101 \
  --batch_size=8 \
  --save_freq=10000 \
  --wandb.enable=true

# SmolVLA fine-tuning launch — v2 dataset (v1 + 16 edge-midpoint episodes),
# trim_frames_v2.json. Identical recipe to the v1 launch; only data changed.
set -e

# v1 freeze was e2dd884ad6676140762e8172e50978c2424bbcc2 (so101_policy_robustness)
DATASET_FREEZE_HASH=801d2458c243b06314b8b43c41ee3bb71c5066aa

python train_policy.py \
  --dataset.repo_id=TamaraSumarac/so101_policy_robustness_v2 \
  --dataset.revision=$DATASET_FREEZE_HASH \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --output_dir=outputs/smolvla_v2 \
  --job_name=smolvla_v2_so101 \
  --batch_size=8 \
  --save_freq=10000 \
  --wandb.enable=true

#!/bin/bash
# 20-trial nominal eval on the frozen 5x4 grid (episodic strategy, recorded).
#
# Usage:
#   bash rollout/eval_nominal.sh diffusion
#   bash rollout/eval_nominal.sh smolvla
#   (act evaluated in Week 7: rollout_act_nominal_eval_20260818_104621)
#
# Protocol (frozen): same 20 start cells in the same order as eval_nominal_act.md;
# 30 s budget; 20 s reset; success = block at rest fully within plate rim;
# ALL trials scored, no discards (left arrow forbidden); right arrow ends an
# episode once the outcome is decided.
# Ritual: leader boxed; camera verified by what it SEES; phone rolling one
# continuous take; eval .md open for live notes.
set -e

EXTRA_ARGS=()
case "$1" in
  diffusion)
    POLICY=checkpoints_diffusion_baseline/100000_ddim10   # DDIM-10: see deployment note in results_diffusion.md
    REPO=TamaraSumarac/rollout_diffusion_nominal_eval ;;
  smolvla)
    POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
    REPO=TamaraSumarac/rollout_smolvla_nominal_eval
    EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}') ;;
  *) echo "usage: bash rollout/eval_nominal.sh {diffusion|smolvla}"; exit 1 ;;
esac

lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B610326031 \
  --robot.id=so101_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.path=$POLICY \
  --strategy.type=episodic \
  --inference.type=sync \
  --device=mps \
  --task="Pick up the pink block and place it in the plate" \
  --dataset.repo_id=$REPO \
  --dataset.single_task="Pick up the pink block and place it in the plate" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --dataset.push_to_hub=false \
  "${EXTRA_ARGS[@]}" \
  --display_data=true

#!/bin/bash
# Base-strategy rollout: policy drives the arm, NOTHING is recorded.
# For shakedowns and curiosity runs only — evals use the episodic strategy.
#
# Usage:
#   bash tools/rollout_base.sh act
#   bash tools/rollout_base.sh diffusion
#   bash tools/rollout_base.sh smolvla
#
# Ritual: leader boxed; camera verified by what it SEES; block placed; hand
# near power. Task string stays verbatim (SmolVLA consumes it as model input).
set -e
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
case "$1" in
  act)       POLICY=checkpoints_act_baseline/100000/pretrained_model ;;
  diffusion) POLICY=checkpoints_diffusion_baseline/100000_ddim10 ;;   # DDIM-10 variant: see results_diffusion.md
  smolvla)   POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
             # smolvla_base declares 3 pretraining camera slots; our single
             # camera maps to camera1, same as at training time.
             EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}') ;;
  *) echo "usage: bash tools/rollout_base.sh {act|diffusion|smolvla}"; exit 1 ;;
esac

lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B610326031 \
  --robot.id=so101_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.path=$POLICY \
  --strategy.type=base \
  --inference.type=sync \
  --device=mps \
  --task="Pick up the pink block and place it in the plate" \
  "${EXTRA_ARGS[@]}" \
  --duration=30 \
  --display_data=true

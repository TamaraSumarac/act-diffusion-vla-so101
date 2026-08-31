#!/bin/bash
# One-episode RTC probe for SmolVLA (episodic strategy, RECORDED).
# Purpose: measure per-step joint deltas under RTC vs demos/sync — NOT a scored eval.
# Episodic pacing also answers whether the RTC loop respects 30 fps here,
# unlike the free-running base-strategy smoke.
#
# Usage: bash tools/eval_smolvla_rtc_test.sh
#
# Safety: this is the forceful-motion regime — hand near power, end the episode
# (right arrow) at any jitter under contact. Calibration gate before and after.
# Episode 0 is still the compile warm-up: empty table, right-arrow it after a
# few seconds; episode 1 is the probe with the block at Center, 0 degrees.
set -e

cd "$(dirname "$0")/.."   # anchor to repo root (script lives in tools/)

POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
REPO=TamaraSumarac/rollout_rtc_probe_smolvla_eh30

lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B610326031 \
  --robot.id=so101_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.path=$POLICY \
  --strategy.type=episodic \
  --inference.type=rtc \
  --inference.rtc.execution_horizon=30 \
  --inference.rtc.debug=true \
  --device=mps \
  --task="Pick up the pink block and place it in the plate" \
  --dataset.repo_id=$REPO \
  --dataset.single_task="Pick up the pink block and place it in the plate" \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --dataset.push_to_hub=false \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --display_data=true
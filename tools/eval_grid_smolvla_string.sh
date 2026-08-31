#!/bin/bash
# Perturbation-grid eval on the frozen start grids (episodic strategy, recorded).
# Perturbations are physical; this script only labels and records.
#
# Usage:
#   bash rollout/eval_grid.sh act position_shift
#   bash rollout/eval_grid.sh smolvla slide
#
# Protocol: perturbation_protocol.md (frozen). 30 s budget; 20 s reset;
# success = block at rest fully within plate rim; ALL trials scored, no
# discards (left arrow forbidden); right arrow ends an episode once decided.
# Ritual: leader boxed; camera verified by what it SEES; calibration gate
# passed; eval .md open for live notes; hand near power for slide trials.
#
# SmolVLA warm-up: kernel compile (~7.6 s) is per PROCESS, so the throwaway
# must happen inside this run. SmolVLA gets one extra episode: episode 0 is
# the warm-up — empty table, let it run a few seconds, right-arrow it, log
# as "void (warm-up)". Scored trials are episodes 1..N. For ACT, episodes
# are 0..N-1, no warm-up needed.
set -e

cd "$(dirname "$0")/.."   # anchor to repo root (script lives in tools/)

# POLICY_NAME=$1
# CONDITION=$2

# case "$POLICY_NAME" in
#   act)     POLICY=checkpoints_act_baseline/100000/pretrained_model ;;
#   smolvla) POLICY=checkpoints_smolvla_baseline/100000/pretrained_model ;;
#   *) echo "usage: bash rollout/eval_grid.sh {act|smolvla} {position_shift|novel_color|distractor|lighting|novel_object|slide}"; exit 1 ;;
# esac

# case "$CONDITION" in
#   position_shift)                                  N_TRIALS=16 ;;
#   novel_color|distractor|lighting|novel_object|slide) N_TRIALS=20 ;;
#   *) echo "usage: bash rollout/eval_grid.sh {act|smolvla} {position_shift|novel_color|distractor|lighting|novel_object|slide}"; exit 1 ;;
# esac

# REPO=TamaraSumarac/rollout_eval_${CONDITION}_${POLICY_NAME}

REPO=TamaraSumarac/rollout_eval_novel_color_smolvla_string_change


EXTRA_ARGS=(--rename_map='{"observation.images.front": "observation.images.camera1"}')
# NUM_EPISODES=$N_TRIALS
# if [ "$POLICY_NAME" = "smolvla" ]; then
#   # smolvla_base declares 3 pretraining camera slots; our single camera maps
#   # to camera1, same as at training time.
#   EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}')
#   NUM_EPISODES=$((N_TRIALS + 1))   # episode 0 = warm-up throwaway (see header)
# fi

# echo "policy=$POLICY_NAME condition=$CONDITION repo=$REPO episodes=$NUM_EPISODES (scored trials: $N_TRIALS)"

lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B610326031 \
  --robot.id=so101_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.path=checkpoints_smolvla_baseline/100000/pretrained_model \
  --strategy.type=episodic \
  --inference.type=sync \
  --device=mps \
  --task="Pick up the light blue block and place it in the plate" \
  --dataset.repo_id=$REPO \
  --dataset.single_task="Pick up the light blue block and place it in the plate" \
  --dataset.num_episodes=21 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --dataset.push_to_hub=false \
  "${EXTRA_ARGS[@]}" \
  --display_data=true
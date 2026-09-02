#!/bin/bash
# 20-trial nominal eval on the frozen 5x4 grid (episodic strategy, recorded).
#
# Usage:
#   bash tools/eval_nominal.sh diffusion
#   bash tools/eval_nominal.sh smolvla
#   bash tools/eval_nominal.sh smolvla_v2_ep66
#   bash tools/eval_nominal.sh smolvla_remote_box    # server on A10 via ssh -L 8080:localhost:8080
#   (act evaluated: rollout_act_nominal_eval_20260818_104621)
#
# Protocol (frozen): same 20 start cells in the same order as eval_nominal_act.md;
# 30 s budget; 20 s reset; success = block at rest fully within plate rim;
# ALL trials scored, no discards (left arrow forbidden); right arrow ends an
# episode once the outcome is decided.
# SmolVLA local runs: 20 episodes, all scored (no warm-up), matching the v1 baseline.
# Remote runs: 21 episodes, episode 0 = warm-up (server kernel compile), scored = 1–20.
# Ritual: leader boxed; camera verified by what it SEES; phone rolling one
# continuous take; eval .md open for live notes.
set -e
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
INFERENCE=sync
NUM_EPISODES=20
case "$1" in
  diffusion)
    POLICY=checkpoints_diffusion_baseline/100000_ddim10   # DDIM-10: see deployment note in results_diffusion.md
    REPO=TamaraSumarac/rollout_diffusion_nominal_eval ;;
  diffusion_remote_box)
    # Frozen 2026/09/02 config: DDIM-50 + clamp 8 + full-chunk sync semantics.
    # See results_diffusion_ctnd.md remote-deployment section. Server must run
    # the 2026/09/02-patched policy_server.py + helpers.py (2-frame history).
    POLICY=checkpoints_diffusion_baseline/100000_ddim10   # local POLICY supplies config shape only (device, action names, visual features); behavioral fields aren't read
    REPO=TamaraSumarac/rollout_diffusion_remote_nominal_eval
    EXTRA_ARGS+=(--robot.max_relative_target=8)
    EXTRA_ARGS+=(--inference.policy_path_on_server=/home/ubuntu/checkpoints_diffusion_ddim50_100k)
    EXTRA_ARGS+=(--inference.policy_device=cuda)
    EXTRA_ARGS+=(--inference.policy_type=diffusion)
    EXTRA_ARGS+=(--inference.actions_per_chunk=32)
    EXTRA_ARGS+=(--inference.image_rename='{}')
    INFERENCE=remote
    NUM_EPISODES=21 ;;
  smolvla)
    POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
    REPO=TamaraSumarac/rollout_smolvla_nominal_eval
    EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}') ;;
  smolvla_v2_ep66)
    POLICY=checkpoints_smolvla_v2_66ep/100000
    REPO=TamaraSumarac/rollout_smolvla_v2_ep66_nominal_eval
    EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}') ;;
  smolvla_remote_box)
    # Fork's remote sync engine; the server loads the checkpoint from the BOX path we send.
    POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
    REPO=TamaraSumarac/rollout_smolvla_remote_nominal_eval
    EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}')
    EXTRA_ARGS+=(--inference.policy_path_on_server=/home/ubuntu/checkpoints_smolvla_baseline_100k)
    EXTRA_ARGS+=(--inference.policy_device=cuda)
    EXTRA_ARGS+=(--inference.policy_type=smolvla)
    EXTRA_ARGS+=(--inference.actions_per_chunk=50)
    EXTRA_ARGS+=(--inference.image_rename='{"observation.images.front": "observation.images.camera1"}')
    INFERENCE=remote
    NUM_EPISODES=21 ;;
  *) echo "usage: bash tools/eval_nominal.sh {diffusion|smolvla|smolvla_v2_ep66|smolvla_remote_box|diffusion_remote_box}"; exit 1 ;;
esac

lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B610326031 \
  --robot.id=so101_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.path=$POLICY \
  --strategy.type=episodic \
  --inference.type=$INFERENCE \
  --device=mps \
  --task="Pick up the pink block and place it in the plate" \
  --dataset.repo_id=$REPO \
  --dataset.single_task="Pick up the pink block and place it in the plate" \
  --dataset.num_episodes=$NUM_EPISODES \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=20 \
  --dataset.push_to_hub=false \
  "${EXTRA_ARGS[@]}" \
  --display_data=true
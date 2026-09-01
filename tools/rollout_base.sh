#!/bin/bash
# Base-strategy rollout: policy drives the arm, NOTHING is recorded.
# For shakedowns and curiosity runs only — evals use the episodic strategy.
#
# Usage:
#   bash tools/rollout_base.sh act
#   bash tools/rollout_base.sh diffusion
#   bash tools/rollout_base.sh smolvla
#   bash tools/rollout_base.sh smolvla_remote_local   # server on Mac: bash tools/async_smoke.sh server
#   bash tools/rollout_base.sh smolvla_remote_box     # server on A10 via ssh -L 8080:localhost:8080
#
# Ritual: leader boxed; camera verified by what it SEES; block placed; hand
# near power. Task string stays verbatim (SmolVLA consumes it as model input).
set -e
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
INFERENCE=sync
case "$1" in
  act)       POLICY=checkpoints_act_baseline/100000/pretrained_model ;;
  diffusion) POLICY=checkpoints_diffusion_baseline/100000_ddim10 ;;   # DDIM-10 variant: see results_diffusion.md
  smolvla)   POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
             # smolvla_base declares 3 pretraining camera slots; our single
             # camera maps to camera1, same as at training time.
             EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}') ;;
  smolvla_rtc)
             POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
             EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}')
             INFERENCE=rtc ;;
  smolvla_remote_local)
             # Fork's remote sync engine, policy server on the Mac (plumbing smoke
             # only; ~1s/chunk on MPS). The server loads the checkpoint from the
             # path we send it, so it must be valid on the server's machine.
             POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
             EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}')
             EXTRA_ARGS+=(--inference.policy_path_on_server="$PWD/$POLICY")
             EXTRA_ARGS+=(--inference.policy_device=mps)
             INFERENCE=remote ;;
  smolvla_remote_box)
             # Policy server on the A10 box, reached through the SSH tunnel.
             # Path + device are the BOX's.
             POLICY=checkpoints_smolvla_baseline/100000/pretrained_model
             EXTRA_ARGS+=(--rename_map='{"observation.images.front": "observation.images.camera1"}')
             EXTRA_ARGS+=(--inference.policy_path_on_server=/home/ubuntu/checkpoints_smolvla_baseline_100k)
             EXTRA_ARGS+=(--inference.policy_device=cuda)
             INFERENCE=remote ;;
  *) echo "usage: bash tools/rollout_base.sh {act|diffusion|smolvla|smolvla_rtc|smolvla_remote_local|smolvla_remote_box}"; exit 1 ;;
esac

lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B610326031 \
  --robot.id=so101_follower \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --policy.path=$POLICY \
  --strategy.type=base \
  --inference.type=$INFERENCE \
  --device=mps \
  --task="Pick up the pink block and place it in the plate" \
  "${EXTRA_ARGS[@]}" \
  --duration=30 \
  --display_data=true
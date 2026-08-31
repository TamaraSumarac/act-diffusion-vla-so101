#!/bin/bash
# Remote-inference smoke: policy server + robot client (lerobot.async_inference).
# Tests whether serving SmolVLA inference off-device cuts per-chunk dead time
# (~1s on MPS vs ~0.24s measured on A10 + ~0.10s network RTT).
# NOTHING is recorded. The arm MOVES when the client connects.
#
# Usage:
#   LOCAL SMOKE (both halves on the Mac — control condition, expect jerky):
#     Terminal 1:  bash tools/async_smoke.sh server
#     Terminal 2:  bash tools/async_smoke.sh client
#
#   REMOTE (server on the A10 box — the experiment):
#     Terminal 1:  ssh -L 8080:localhost:8080 ubuntu@<BOX_IP>     # tunnel, keep open
#     ...then inside that ssh session, ON THE BOX:
#                  python -m lerobot.async_inference.policy_server --host localhost --port 8080 --fps 30
#     Terminal 2:  bash tools/async_smoke.sh client_box
#
# Result log (local smoke): pipe works end-to-end, but motion is very jerky —
#   consistent with staleness-ratio diagnosis (~1s MPS inference vs ~1.7s chunk:
#   chunks ~60% expired on arrival). Remote run tests whether dropping the ratio
#   to ~20% (0.24s + 0.10s RTT) restores smooth motion. Same client code both
#   ways; only the staleness ratio changes.
#
# Notes:
# - Camera is declared as "camera1" (NOT "front"): the async path's
#   prepare_raw_observation() indexes policy_image_features by the robot's
#   camera key directly — there is no rename layer on this path, unlike
#   lerobot-rollout where the checkpoint preprocessor's rename_map bridges
#   front -> camera1.
# - The checkpoint preprocessor still contains the front->camera1 rename step;
#   it tolerated the missing "front" key in the local smoke (no patch needed).
# - client_box sends the BOX-side checkpoint path and cuda device to the server;
#   the client itself loads no policy.
# - Ritual unchanged: leader boxed; camera verified by what it SEES; block
#   placed; hand near power. Task string stays verbatim.
set -e
cd "$(dirname "$0")/.."

CKPT="checkpoints_smolvla_baseline/100000/pretrained_model"          # Mac path (local smoke)
CKPT_BOX="/home/ubuntu/checkpoints_smolvla_baseline_100k"            # box path (client_box)
TASK="Pick up the pink block and place it in the plate"
PORT=8080

case "$1" in
  server)
    python -m lerobot.async_inference.policy_server \
      --host localhost \
      --port $PORT \
      --fps 30
    ;;
  client)
    SERVER_ADDRESS=${2:-localhost:$PORT}
    python -m lerobot.async_inference.robot_client \
      --robot.type=so101_follower \
      --robot.id=so101_follower \
      --robot.port=/dev/tty.usbmodem5B610326031 \
      --robot.cameras="{camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
      --policy_type=smolvla \
      --pretrained_name_or_path="$CKPT" \
      --policy_device=mps \
      --task="$TASK" \
      --server_address=$SERVER_ADDRESS \
      --actions_per_chunk=50 \
      --fps=30
    ;;
  client_box)
    # Requires the tunnel (Terminal 1) and the server running on the box.
    SERVER_ADDRESS=${2:-localhost:$PORT}
    python -m lerobot.async_inference.robot_client \
      --robot.type=so101_follower \
      --robot.id=so101_follower \
      --robot.port=/dev/tty.usbmodem5B610326031 \
      --robot.cameras="{camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
      --policy_type=smolvla \
      --pretrained_name_or_path="$CKPT_BOX" \
      --policy_device=cuda \
      --task="$TASK" \
      --server_address=$SERVER_ADDRESS \
      --actions_per_chunk=50 \
      --fps=30 \
      "${@:3}"
    ;;
  *) echo "usage: bash tools/async_smoke.sh {server|client|client_box} [server_address]"; exit 1 ;;
esac
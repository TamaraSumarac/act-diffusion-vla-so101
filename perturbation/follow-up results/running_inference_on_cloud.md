# Running SmolVLA inference on a cloud GPU (remote policy server)

## Why
SmolVLA's per-chunk inference on the MacBook (MPS) takes ~1s against a ~1.7s
action chunk. Under sync inference this is dead time the arm spends waiting; under
RTC/async it becomes staleness (chunks ~60% expired on arrival → jerky motion).
Trace analysis showed SmolVLA makes ~1.25 grasp attempts/episode vs ACT's ~2.25
but converts better per attempt (0.28 vs 0.20). Hypothesis: inference latency
is SmolVLA's binding constraint; a faster brain should raise attempts/episode and
nominal success toward ~60%.

Prior attempt (RTC on the Mac) failed because the *ratio* of inference time to
chunk length was too high — not because async is wrong per se.

## Gates (measure before building)

| Gate | Test | Result |
|---|---|---|
| 0 | Bare GPU chunk inference, A10, dummy obs (`tools/box_inference_check.py`) | **0.24s median** (3ms spread) vs ~1s on MPS |
| 1 | Network round trip, 100KB payload over persistent SSH | **~0.10s** steady-state (0.41s with per-call handshake — irrelevant, server holds one connection) |

Budget: 0.24 + 0.10 ≈ **0.34s per chunk** vs ~1s local. Staleness ratio
0.34/1.7 ≈ 20% (vs ~60% that broke RTC locally).

## Architecture
- **Box (A10):** `python -m lerobot.async_inference.policy_server --host localhost --port 8080 --fps 30`
- **Mac:** `python -m lerobot.async_inference.robot_client ...` — owns the arm + camera, ships raw observations, executes returned chunks.
- **Transport:** SSH tunnel `ssh -L 8080:localhost:8080 ubuntu@<BOX_IP>` — no firewall config, and it's exactly what Gate 1 measured.
- The client sends `policy_type` + `pretrained_name_or_path` to the server, which loads the policy. **The checkpoint path must resolve on the box.**
- `lerobot.async_inference` exists in the fork; needed only `pip install grpcio grpcio-tools` on both machines.
- Script: `tools/async_smoke.sh {server|client|client_box}` with extra-arg pass-through.

## Bugs found and fixed, in order

### 1. Camera key mismatch (server KeyError `observation.images.front`)
`helpers.prepare_raw_observation()` indexes `policy_image_features` by the
robot's camera key directly — no rename layer. The sync path bridges
`front → camera1` via the checkpoint preprocessor's `rename_map`; the async
path does not. **Fix:** declare the camera as `camera1` in `--robot.cameras`.
The preprocessor's rename step tolerates the missing `front` key.

### 2. Jerky motion in local smoke (both halves on Mac)
Expected and diagnostic: same client with ~1s inference reproduces the RTC
staleness pathology. Moving the server to the box → **smooth motion, same
client code**. Confirms the staleness-ratio theory on hardware.

### 3. Systematic lateral grasp offset on the remote path (smooth but misses)
Sync rollout grabbed center-0° the same day → rig/camera/checkpoint fine;
offset was specific to the remote path. Root cause in
`helpers.resize_robot_observation_image()` (runs **server-side**):

- **v0 (upstream):** plain bilinear interpolate 640×480 → 256×256 — aspect
  squash, ~1.33× horizontal compression. Training/sync feed raw 640×480 to
  SmolVLA's internal aspect-preserving `resize_with_pad` (512×512, left/top
  pad). → Large offset.
- **v1:** replaced with `resize_with_pad` to 256×256 (aspect-correct). →
  Smaller offset, still missed center-0°. Double resize (256 then 512
  internally) still shifts effective geometry / loses resolution.
- **v2 (final):** no resize at all — permute + uint8→[0,1] only; let the model
  do the single resize it trained with, exactly as sync does. → **Grabs
  center-0°, matches same-day sync behavior.**

Lesson: match the training-time image path *exactly*; even an aspect-correct
extra resize is enough to miss. Value range verified: model expects [0,1]
(does `*2-1` internally), preprocessor visual norm is IDENTITY, no double-scaling.

**Deployment note:** `helpers.py` is imported by both halves, and the resize
runs on the server. The fix must live in the **box's venv copy** — the first
attempt patched only the Mac fork and changed nothing. Recovery on any fresh
box: `scp ~/lerobot/src/lerobot/async_inference/helpers.py ubuntu@<BOX_IP>:/home/ubuntu/venv/lib/python3.12/site-packages/lerobot/async_inference/helpers.py`
then restart the server.

### 4. Tokenization (benchmark only)
`box_inference_check.py` bypasses the preprocessor, so it tokenizes the task
manually. The async server runs the full checkpoint preprocessor (tokenizer
step included) — not a factor for the offset.

## Fork modifications (Mac, `~/lerobot/src/lerobot/async_inference/`)
- `configs.py` — `RobotClientConfig`: added `duration: float | None` and
  `display_data: bool` (placed last; dataclass default-ordering).
- `robot_client.py` — Rerun logging of camera frames when `display_data`;
  duration-based clean exit via `shutdown_event.set()` (`running` is a
  read-only property).
- `helpers.py` — `resize_robot_observation_image` v2 pass-through (also
  scp'd to the box).

## `chunk_size_threshold` bracket (fraction of chunk remaining when re-planning)
| Value | Behavior |
|---|---|
| 0.0 | Pure sync: ~0.35s freeze per chunk, lurchy. Grabs. |
| 0.1 | Slight freeze (5 steps ≈ 0.17s < 0.35s latency). |
| **0.2** | **Best.** New chunk lands as queue empties — no freeze, minimal overlap. Matches prediction 0.35s × 30fps / 50 ≈ 0.2. |
| 0.3 | Offset creeps back (more overlap → `weighted_average` blends stale plan in). |

Eval config: **0.2**.

## Banked numbers
- GPU inference: 0.24s/chunk (real observations, A10)
- Network: ~0.10s round trip
- Remote chunk budget: ~0.34s vs ~1s local
- Center-0° grasp: restored on remote path, matches same-day sync

## Open items
- **Recording.** The async client records nothing; the eval needs datasets for
  trace analysis. Options: (a) add recording to the async client; (b) add a
  `remote` engine to the fork's `--inference.type` slot so `lerobot-rollout`
  (episodic strategy, recording, warm-up convention) talks to the box;
  (c) hand-score + video for a first pass. Decision pending.
- Then: 20-trial SmolVLA nominal eval on the remote path. Prediction on
  record: attempts/episode 1.25 → ~2.2; success toward ~60% at 0.28
  conversion. All three outcomes informative.
- Box setup for next rental: `pip install grpcio grpcio-tools`, `hf download`
  the checkpoint, scp patched `helpers.py`, tunnel.
# Running SmolVLA inference on a cloud GPU (remote policy server 2026/08/31)

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


---

# Part 2 — remote *sync* inference inside `lerobot-rollout` (2026/09/01)

## Decision: recording via a `remote` inference engine (option b)
The async client was validated and bracketed (Part 1) but records nothing and is async by construction. For the eval I wanted pure sync semantics — request a chunk, block until it arrives, execute all 50 actions, request the next — so the *only* variable vs. the local nominal run is dead time per chunk. Cleanest place for that is the fork's `--inference.type` slot: `lerobot-rollout` keeps the
episodic strategy, recording, end-episode key and warm-up convention unchanged.

## Fork modifications (`~/lerobot/src/lerobot/rollout/inference/`)
- `remote.py` (new) — `RemoteSyncInferenceEngine`. Owns a gRPC stub to the upstream `policy_server`; per `get_action` call pops from a local FIFO, and when empty sends the observation (`must_go=True`) and blocks on `GetActions`. Inverts `build_dataset_frame` to rebuild the raw robot obs the server expects (state via `observation.state["names"]`, camera renamed `front → camera1` on the wire). Returned actions are already postprocessed server-side; only reordered to `ordered_action_keys` for parity with `SyncInferenceEngine`. Logs median round trip on stop. The locally loaded policy is unused.
- `factory.py` — `RemoteInferenceConfig` registered as `"remote"` (`server_address`, `policy_type`, `policy_path_on_server`, `policy_device`, `actions_per_chunk`, `image_rename`, `chunk_timeout_s`) + dispatch.
- `async_inference/helpers.py` — v2 pass-through resize consolidated in place (previous version had the upstream def and the override coexisting; last definition wins in Python, but it was misleading).

## Repo scripts
- `tools/rollout_base.sh` — cases `smolvla_remote_local` (server on Mac, plumbing smoke) and `smolvla_remote_box` (server on A10 via tunnel).
- `tools/eval_nominal.sh` — case `smolvla_remote_box`; `INFERENCE` and `NUM_EPISODES` are now variables (was hardcoded `sync` / `20`); 21 episodes,
  episode 0 = warm-up.
- `tools/async_smoke.sh` — still the way to launch the server (`server` subcommand); the `client*` cases are the Part 1 async path, kept as record.

## Validation sequence (all passed first try)
1. `rollout_base.sh smolvla_remote_local` — chunks flow, arm task-shaped.
2. Episodic 2-episode recording smoke against the Mac server — dataset written, `lerobot-dataset-viz` plays back.
3. Box: `rollout_base.sh smolvla_remote_box` — center-0° grab, server log shows observation timesteps 300 → 350 → 400 with `must_go: True` (one request per 50-action drain, no overlap), ~0.23 s server-side per chunk.

## Box setup (fresh rental)
```bash
scp training/setup_box.sh ubuntu@<IP>:~/ && ssh ubuntu@<IP> "bash setup_box.sh"
ssh ubuntu@<IP> "source ~/venv/bin/activate && pip install 'lerobot[smolvla]' grpcio grpcio-tools"
rsync -avz --progress checkpoints_smolvla_baseline/100000/pretrained_model/ ubuntu@<IP>:~/checkpoints_smolvla_baseline_100k/
scp ~/lerobot/src/lerobot/async_inference/helpers.py ubuntu@<IP>:~/venv/lib/python3.12/site-packages/lerobot/async_inference/helpers.py
ssh -L 8080:localhost:8080 ubuntu@<IP>        # keep open; on the box:
source ~/venv/bin/activate && python -m lerobot.async_inference.policy_server --host localhost --port 8080 --fps 30
```
Gotchas: fresh box lacked `transformers` (`lerobot[smolvla]` extra — add to `setup_box.sh`); verify the `helpers.py` patch survived any pip install (`grep -n "v2" .../async_inference/helpers.py`).

## Result (details + table in `smolvla_remote_inference.md`)
- End-to-end round trip: **median 0.48 s/chunk** over 261 chunks (0.24 s inference + ~0.24 s transport; the raw 640×480 frame is ~920 KB pickled, 9× the Gate 1 payload) vs ~1 s local.
- Attempts/ep **1.25 → 1.60**, success **35% → 30%** (flat), conversion 0.28 → 0.19.
- Why the 2.2 prediction was unreachable: chunk cycle = execution + dead time. Local 1.67 + 1.0 = 2.67 s, remote 1.67 + 0.48 = 2.15 s → +24% chunks/s, and attempts rose 18–28%. The lever was right, the throw was bounded by execution time. Latency moves attempts; data coverage moves conversion (v2: 0.37, 55%).

## Open items / next
- **Diffusion via remote inference.** Server calls `predict_action_chunk`, not `select_action`, so diffusion's `n_obs_steps=2` history queues never fill and it falls into the single-frame offline branch. Needs a small server patch: `populate_queues` before the chunk call, and `policy.reset()` when an observation arrives with timestep 0 (the remote engine restarts timestep on episode reset). Plus `--inference.policy_type=diffusion`, `--inference.image_rename='{}'`. With 8-action chunks the ~0.24 s transport dominates — JPEG on the wire becomes necessary, not optional. GPU also makes more denoising steps affordable (DDIM-10 was a Mac-speed compromise).
- Shorter executed chunks (`--inference.actions_per_chunk=25`) is *not* a throughput lever (moving fraction 78% → 63%) but is a reactivity/conversion hypothesis: re-observe every 0.83 s and correct the reach. Untested.
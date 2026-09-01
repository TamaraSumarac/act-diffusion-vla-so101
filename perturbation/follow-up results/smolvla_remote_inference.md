# SmolVLA trained on old data but inference run on remote powerful GPU (nominal test - 20 trials)

- **Date:** 2026-09-01
- **Policy:** SmolVLA, checkpoint 100000 (`checkpoints_smolvla_baseline/100000/pretrained_model`)
- **Dataset trained on:** `TamaraSumarac/so101_policy_robustness` @ `e2dd884` (trimmed, 20975 frames)
- **Protocol:** fixed 5-position × 4-orientation grid (frozen for all policies); binary success = block at rest fully within plate rim; all trials scored, no discards; episode budget 30 s
- **Rollout dataset:** `TamaraSumarac/rollout_smolvla_remote_nominal_eval` (local only)
- **Episode mapping:** episode 0 = warm-up (void), scored trials = episodes 1–20
- **Block orientation:** pink faces up, yellow faces toward window at 0° (matches standard block placement)
- **Deployment:** `lerobot-rollout`, strategy=episodic, inference=remote (sync semantics: one blocking chunk request per 50-action drain), policy server on cloud A10 (cuda) via SSH tunnel. Measured end-to-end: median 0.48 s per chunk round trip (0.24 s server-side inference + ~0.24 s transport/serialization of the raw 640×480 frame) vs ~1 s per chunk on MPS. Same checkpoint, same protocol as the local nominal run — dead time per chunk is the only variable.

| Trial | Start cell                                   | Success (0/1) | Failure note (one phrase)                                                        |
|-------|----------------------------------------------|---------------|----------------------------------------------------------------------------------|
| 1     | Center, yellow side window, 0 degrees        | 1             |                                                                                  |
| 2     | Top left, yellow side window, 0 degrees      | 0             | tipped it on first try a bit out of the region                                   |
| 3     | Top right, yellow side window, 0 degrees     | 0             | positionally slightly off, was very close on 2nd try                             |
| 4     | Bottom left, yellow side window, 0 degrees   | 0             | positionally slightly off, tipped it multiple times                              |
| 5     | Bottom right, yellow side window, 0 degrees  | 0             | positionally slightly off, i believe this is a consequence of data coverage      |
| 6     | Center, yellow side window, 45 degrees       | 1             | tipped it on pickup                                                              |
| 7     | Top left, yellow side window, 45 degrees     | 0             | positionally off                                                                 |
| 8     | Top right, yellow side window, 45 degrees    | 0             | was positionally slightly off almost succeded on 2nd attempt                     |
| 9     | Bottom left, yellow side window, 45 degrees  | 1             |                                                                                  |
| 10    | Bottom right, yellow side window, 45 degrees | 0             | positionally slightly off, it is trying next the block here                      |
| 11    | Center, yellow side mirror, 90 degrees       | 1             |                                                                                  |
| 12    | Top left, yellow side mirror, 90 degrees     | 0             | positionally off, but next to the block                                          |
| 13    | Top right, yellow side mirror, 90 degrees    | 0             | same as previous one                                                             |
| 14    | Bottom left, yellow side mirror, 90 degrees  | 0             | same as previous one                                                             |
| 15    | Bottom right, yellow side mirror, 90 degrees | 0             | same as previous one                                                             |
| 16    | Center, yellow side door, 135 degrees        | 1             |                                                                                  |
| 17    | Top left, yellow side door, 135 degrees      | 0             | tipped it to top left 90 deg on first attempt                                    |
| 18    | Top right, yellow side door, 135 degrees     | 0             | pushed it more to the top on first attempt and wrist angle was worse             |
| 19    | Bottom left, yellow side door, 135 degrees   | 0             | tipped it on 1st try, 2nd try next to block and wrong angle                      |
| 20    | Bottom right, yellow side door, 135 degrees  | 1             |                                                                                  |

## Result

- **Successes:**  6 / 20
- **Nominal success rate:**  30%

## Results: attempts vs. success
- Analysis in perturbation/DataAnalysis.ipynb

Same v1 checkpoint, same 20-cell grid, same protocol. Only difference is where inference runs: MPS on the Mac (~1 s per chunk) vs. A10 on the cloud (0.48 s per chunk end-to-end). Note, v2 (66-demo checkpoint, local MPS) is added as a reference point, but we are not testing it for latency.

| Run       | Inference          | Attempts/ep | Episode (s) | Success | Conversion (success/attempt) |
|-----------|--------------------|-------------|-------------|---------|------------------------------|
| v1 local  | MPS, ~1.0 s/chunk  | 1.25        | 18.7        | 35%     | 0.28                         |
| v1 remote | A10, 0.48 s/chunk  | 1.60        | 20.2        | 30%     | 0.19                         |
| v2 local  | MPS, ~1.0 s/chunk  | 1.50        | 18.8        | 55%     | 0.37                         |

Attempts counted from the shoulder_pan / gripper trace (visit to the start region with a gripper close), same script as the robustness study. v1 remote skips episode 0 (warm-up).

### What this says

- **Faster inference bought more attempts, but only ~25%, not the ~75% we naively thought.** Reason is simple once you write out the chunk cycle: execution + dead time. Local is 1.67 + 1.0 = 2.67 s per chunk, remote is 1.67 + 0.48 = 2.15 s. That's 24% more chunks per second, and attempts went up 18–28% (per second / per episode). So the mechanism was right, the magnitude wasn't — with a 50-action chunk, execution time dominates the cycle and halving dead time can't do more than that. Getting to ACT's 2.25 attempts/ep was never a latency question.
- **Success didn't move (35% → 30%).** More attempts, same misses — the failure notes are almost all "positionally slightly off". Conversion per attempt actually dropped (0.28 → 0.19). Latency was limiting how often the model tries, not how precisely it grasps.
- **v2 shows what precision looks like.** Conversion 0.37, success 55%. That came from 16 more demos, not from speed. (Interestingly v2 also tries more, 1.50/ep.)

So the two levers are separate: latency moves attempts (bounded by chunk execution time), data coverage moves conversion, and success ≈ attempts × conversion. Neither replaces the other.


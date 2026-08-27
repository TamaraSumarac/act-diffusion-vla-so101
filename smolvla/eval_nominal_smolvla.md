# SmolVLA — Nominal Eval (20 trials)

- **Date:** 2026-08-24
- **Policy:** SmolVLA, checkpoint 100000, fine-tuned from lerobot/smolvla_base (`checkpoints_smolvla_baseline/100000/pretrained_model`)
- **Dataset trained on:** `TamaraSumarac/so101_policy_robustness` @ `e2dd884` (trimmed; sampler covered 20975 frames (no preset frame drop))
- **Deployment:** `lerobot-rollout`, strategy=episodic, inference=sync, device=mps, 30 fps; ~1.0 s per 50-action chunk; single camera mapped to camera1 slot via rename_map (as at training); task string verbatim from training
- **Protocol:** frozen 5-position × 4-orientation grid, identical cells and order to eval_nominal_act.md; binary success = block at rest fully within plate rim; all trials scored, no discards; 30 s budget
- **Rollout dataset:** `TamaraSumarac/rollout_smolvla_nominal_eval` (local only)
- All blocks start with pink side of the block up

| Trial | Start cell                                   | Success (0/1) | Failure note (one phrase)                                                                      |
|-------|----------------------------------------------|---------------|------------------------------------------------------------------------------------------------|
| 1     | Center, yellow side window, 0 degrees        | 1             |                                                                                                |
| 2     | Top left, yellow side window, 0 degrees      | 1             | dragged it to the middle on first attempt                                                      |
| 3     | Top right, yellow side window, 0 degrees     | 0             | position roughly right but slightly off; attempted pickup with gripper at middle               |
| 4     | Bottom left, yellow side window, 0 degrees   | 1             | dragged and pushed it to middle, tipped it to pick it up                                       |
| 5     | Bottom right, yellow side window, 0 degrees  | 0             | position again biased toward center; gripper pushed the block, couldn't pick it up             |
| 6     | Center, yellow side window, 90 degrees       | 1             |                                                                                                |
| 7     | Top left, yellow side window, 90 degrees     | 0             | angle and position off from the start (going for the center)                                   |
| 8     | Top right, yellow side window, 90 degrees    | 0             | same: default position, tilted gripper, center (typical from training)                         |
| 9     | Bottom left, yellow side window, 90 degrees  | 0             | going for the center, gripper at the 0-degree angle                                            |
| 10    | Bottom right, yellow side window, 90 degrees | 0             | same as rest of 90-degree row                                                                  |
| 11    | Center, yellow side mirror, 45 degrees       | 1             |                                                                                                |
| 12    | Top left, yellow side mirror, 45 degrees     | 0             | tipped it on first attempt and the following one                                               |
| 13    | Top right, yellow side mirror, 45 degrees    | 0             | picked up block on last attempt but didn't drop it                                             |
| 14    | Bottom left, yellow side mirror, 45 degrees  | 0             | position slightly off; pushed block toward 90 degrees on first attempt, hard to recover        |
| 15    | Bottom right, yellow side mirror, 45 degrees | 0             | same as previous one                                                                           |
| 16    | Center, yellow side door, 135 degrees        | 1             |                                                                                                |
| 17    | Top left, yellow side door, 135 degrees      | 0             | RE-RUN: initially staged at 45 degrees by mistake (failed; consistent with trial 12); re-run at correct 135 below - position slighrtly off, pushed block to 90 degrees at first attempt and out of 90 degrees on 2nd |
| 18    | Top right, yellow side door, 135 degrees     | 0             | pushed the block into 90 degrees on first attempt                                              |
| 19    | Bottom left, yellow side door, 135 degrees   | 0             |                                                                                                |
| 20    | Bottom right, yellow side door, 135 degrees  | 1             |                                                                                                |

## Result

- **Successes:** 7 / 20
- **Nominal success rate:** 35 %

## Observations (free-form, post-eval)
- **Position aiming is better than ACT's, but orientation handling is worse.** SmolVLA's
  positions are closer to the true block position compared to ACT (less "going for the middle" on
  approach than ACT at comparable cells), but its wrist-rotation adjustment is weaker:
  in several failures it pushed a block it could have grasped had the gripper angle
  matched the block's orientation. The two policies fail on different axes: ACT
  predominantly on position (reach toward the training mode), SmolVLA predominantly
  on orientation.
- **The 90-degree row failed 1/5 with the same center-seeking signature as ACT's worst
  row.** The pretrained visual prior did not dissolve the mode-seeking failure at the
  hardest orientation: off-center 90-degree placements still pulled approaches toward
  the workspace middle with a 0-degree-style gripper angle.
- **Retry behavior is tighter than ACT's.** After a failed grasp, SmolVLA re-attempts
  locally and immediately (small adjust-and-regrasp cycles) rather than ACT's pattern
  of full retreat-and-reapproach rounds.
- **A new success mode: drag-then-grasp.** In two successes (trials 2, 4) the policy
  dragged/pushed the block toward the workspace middle and grasped it there — reaching
  the training mode by moving the block to it rather than failing away from it. ACT
  never produced this behavior.
- **A new failure mode: grasp-without-release** (trial 13): block picked up on the
  final attempt but never released over the plate. 
- **Deployment texture:** visible idle moments at chunk boundaries (~1 s think-pauses,
  consistent with the measured 1039 ms per 50-action chunk), but motion within chunks
  is smooth — no diffusion-style jitter. Mechanically benign across all 20 trials.

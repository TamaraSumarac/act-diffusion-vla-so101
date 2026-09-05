# Diffusion Baseline — Nominal Eval (20 trials)

- **Date:** 2026-09-02
- **Policy:** Diffusion, checkpoint 100000, DDIM-50 sampling variant (config.json surgery on `checkpoints_diffusion_baseline/100000`; served from `/home/ubuntu/checkpoints_diffusion_ddim50_100k`
  on the A10 box — same weights, `noise_scheduler_type=DDIM`, `num_inference_steps=50`)
- **Dataset trained on:** `TamaraSumarac/so101_policy_robustness` @ `e2dd884` (trimmed, 20975 frames)
- **Deployment:** `lerobot-rollout`, strategy=episodic, inference=remote (fork's sync-semantics engine, A10 via SSH tunnel, ~0.43 s/chunk), 30 fps, full-chunk execution (32 actions/chunk)
- **Diffusion-specific deployment requirements** (see results_diffusion_ctnd.md): `--robot.max_relative_target=8` (clamp; ACT/SmolVLA run unclamped), client-paired 2-frame observation history (1/30 s spacing), JPEG-90 image transport
- **Protocol:** fixed 5-position × 4-orientation grid (frozen for all policies); binary success = block at rest fully within plate rim; all trials scored, no discards; episode budget 30 s;
  21 episodes recorded, episode 0 = warm-up (unscored), scored = 1–20
- **Rollout dataset:** `TamaraSumarac/rollout_diffusion_remote_nominal_eval_<timestamp>` (local only)
- All blocks start with pink side of the block up

| Trial | Start cell                                   | Success (0/1) | Failure note (one phrase)                                                                        |
|-------|----------------------------------------------|---------------|--------------------------------------------------------------------------------------------------|
| 1     | Center, yellow side window, 0 degrees        | 1             |                                                                                                  |
| 2     | Top left, yellow side window, 0 degrees      | 0             | pushed the box around and tipped it on 1st and 2nd try; didn't start from standard start (leftover from previous run) |
| 3     | Top right, yellow side window, 0 degrees     | 0             | almost got it — position quite close                                                             |
| 4     | Bottom left, yellow side window, 0 degrees   | 0             | started frozen from previous run's end pose                                                      |
| 5     | Bottom right, yellow side window, 0 degrees  | 0             | again started where previous episode finished                                                    |
| 6     | Center, yellow side window, 45 degrees       | 1             |                                                                                                  |
| 7     | Top left, yellow side window, 45 degrees     | 0             | started a bit off (end of previous run), tried next to block, pushed the block                   |
| 8     | Top right, yellow side window, 45 degrees    | 0             | position not far off but never tried to grab                                                     |
| 9     | Bottom left, yellow side window, 45 degrees  | 1             | tipped it on pickup                                                                              |
| 10    | Bottom right, yellow side window, 45 degrees | 0             | took a long time for arm to get up; never got there                                              |
| 11    | Center, yellow side mirror, 90 degrees       | 0             | right position but never went for the grab, just oscillated there                                |
| 12    | Top left, yellow side mirror, 90 degrees     | 0             | went for the middle of the start range, close to block                                           |
| 13    | Top right, yellow side mirror, 90 degrees    | 0             | pushed the block a bit, similar to previous                                                      |
| 14    | Bottom left, yellow side mirror, 90 degrees  | 0             | arm struggled to get up, kept trying the whole episode                                           |
| 15    | Bottom right, yellow side mirror, 90 degrees | 0             | same as previous                                                                                 |
| 16    | Center, yellow side door, 135 degrees        | 0             | struggled to get up again even though block is center                                            |
| 17    | Top left, yellow side door, 135 degrees      | 0             | close to block first time but pushed it toward top-left 90-degree position                       |
| 18    | Top right, yellow side door, 135 degrees     | 0             | pushed block around, angle a bit off; started from previous episode's end pose                   |
| 19    | Bottom left, yellow side door, 135 degrees   | 0             | started from previous episode's end pose; position good, angle wrong                             |
| 20    | Bottom right, yellow side door, 135 degrees  | 0             | same as above — position good, angle wrong; started from previous episode's end pose             |

## Result

- **Successes:**  3 / 20
- **Nominal success rate:** 15%

## Pilot invalidated: homing failure under the clamp
The first nominal run (2026-09-02, 3:20pm) is demoted to **pilot**: ~7/20 episodes did not start from the standard start pose — the arm began where the previous episode ended, so those trials measured a different condition than the other policies' evals.

**Diagnosis.** The between episode homing (`_return_to_initial_position`) interpolates current → start over a fixed clock, then stops. With `max_relative_target=8` — the clamp Diffusion requires — every command is re-anchored to the arm's *actual* position, and the clamp-warning log showed the arm falling ~13° behind the interpolation clock over the 1 s homing window. When the loop expired, the last command stopped short of home, and the arm stayed there. Unclamped evals (ACT, SmolVLA) never showed this because the final command equals the true target and the motors drive to their last goal position on their own — the homing routine was always racing the motor; the clamp just removed the safety net that hid it.

**Fix** (`rollout/strategies/core.py`, `episodic.py`):
- Homing (_return_to_initial_position) originally consisted of a single ramp, interpolating the arm from its current pose to the start pose over a fixed 1 s clock. 
- Our fix is consisting of 2 parts - extended the ramp to 3 s — under the clamp the arm fell behind the interpolation and never fully returned.
- But to make sure there are no issues in case we modify clamping position, we added additional settle stage, which after the ramp re-asserts the start pose until every joint is within 2° of target (5 s timeout), so a clock expiry can no longer be an issue. Each reset is verified against the homing log — Homing: 6/6 keys matched, one key per actuator. 
- Verified in following test - position block at the location where arm will be extended 20-30° at the end of the episode, with the fix above arm parks at standard start every reset. 

## New data taking with homing correction
- Clean 20-trial eval below is the scored Diffusion nominal
- Pilot table is saved for the failure taxonomy (its failure to launch and wrong angle clusters are policy-real) and for fun.

| Trial | Start cell                                   | Success (0/1) | Failure note (one phrase) |
|-------|----------------------------------------------|---------------|---------------------------|
| 1     | Center, yellow side window, 0 degrees        | 0.5           | picked block up but episode ended before drop-off; very slow to get arm up |
| 2     | Top left, yellow side window, 0 degrees      | 0             | close initially but tipped block a bit out of range on first try |
| 3     | Top right, yellow side window, 0 degrees     | 0             | arm never got up the whole episode |
| 4     | Bottom left, yellow side window, 0 degrees   | 0             | very close, but arm took too long to get up to engage in time |
| 5     | Bottom right, yellow side window, 0 degrees  | 0             | arm never got up from start (trying the whole episode) |
| 6     | Center, yellow side window, 45 degrees       | 0.5           | picked block up but episode ended before drop-off; very slow to get arm up |
| 7     | Top left, yellow side window, 45 degrees     | 0             | trying close to the block |
| 8     | Top right, yellow side window, 45 degrees    | 0.5           | picked up, nearly delivered, episode ended first; very slow to get arm up |
| 9     | Bottom left, yellow side window, 45 degrees  | 1             |                           |
| 10    | Bottom right, yellow side window, 45 degrees | 0             | arm never got up; almost succeeded rising right at episode end |
| 11    | Center, yellow side mirror, 90 degrees       | 0             | position correct but slow descent; gripper slightly off at grab |
| 12    | Top left, yellow side mirror, 90 degrees     | 0             | arm never got up, kept trying |
| 13    | Top right, yellow side mirror, 90 degrees    | 0             | same as previous |
| 14    | Bottom left, yellow side mirror, 90 degrees  | 0             | slow to get up; once up, tried close to block (not demo-middle) |
| 15    | Bottom right, yellow side mirror, 90 degrees | 0             | arm never got up, trying the whole episode |
| 16    | Center, yellow side door, 135 degrees        | 0             | slow to get up; once up, on the block and grabbing correctly — out of time |
| 17    | Top left, yellow side door, 135 degrees      | 0             | pushed block toward top-left 90° on attempt; wrist adapted well to orientation changes |
| 18    | Top right, yellow side door, 135 degrees     | 0             | tipped block a bit and pushed it around |
| 19    | Bottom left, yellow side door, 135 degrees   | 1             |                           |
| 20    | Bottom right, yellow side door, 135 degrees  | 0             | arm never got up, trying but never succeeded |

## Result

- **Successes:**   5 / 20 (note this is for pickup) 2/20 (for pickup and drop-off)
- **Nominal success rate:** 25% (note this is for pickup) 10% (for pickup and drop-off)

## Observations
- Failures are dominated by a single mechanism: liftoff through the clamp. 6/20 episodes the arm never got up at all; ~7 more spent significant time of the 30 s budget rising (half the time). The clamp that makes diffusion deployable also attenuates its rise phase into the episode clock.
- Once up, the policy is performing well: 5 pickups (3 out of time before drop off), correct on block grasp positionings, and visible wrist adaptation to block orientation (trial 17) suggest that manipulation is not the bottleneck but the arm rise time is.
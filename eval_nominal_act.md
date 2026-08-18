# ACT Baseline — Nominal Eval (20 trials)

- **Date:** 2026-08-18
- **Policy:** ACT, checkpoint 100000 (`checkpoints_act_baseline/100000/pretrained_model`)
- **Dataset trained on:** `TamaraSumarac/so101_policy_robustness` @ `e2dd884` (trimmed, 20975 frames)
- **Deployment:** `lerobot-rollout`, strategy=episodic, inference=sync, device=mps, 30 fps
- **Protocol:** fixed 5-position × 4-orientation grid (frozen for all policies); binary success = block at rest fully within plate rim; all trials scored, no discards; episode budget 30 s
- **Rollout dataset:** `TamaraSumarac/rollout_act_nominal_eval` (local only)
- All blocks start with pink side of the block up

| Trial | Start cell                                   | Success (0/1) | Failure note (one phrase)                                                                    |
|-------|----------------------------------------------|---------------|----------------------------------------------------------------------------------------------|
| 1     | Center, yellow side window, 0 degrees        | 1             |                                                                                                |
| 2     | Top left, yellow side window, 0 degrees      | 1             |                                                                                                |
| 3     | Top right, yellow side window, 0 degrees     | 0             | attempted to grab 3 times, touched last time                                                  |
| 4     | Bottom left, yellow side window, 0 degrees   | 1             | tipped block sideways                                                                          |
| 5     | Bottom right, yellow side window, 0 degrees  | 0             | attempted to grab 3 times, tipped 3rd time                                                    |
| 6     | Center, yellow side window, 45 degrees       | 1             |                                                                                                |
| 7     | Top left, yellow side window, 45 degrees     | 0             | attempted to grab 3 times, touched top first time only                                        |
| 8     | Top right, yellow side window, 45 degrees    | 0             | tipped it with gripper on a side and pushed it out, 3 attempts                                 |
| 9     | Bottom left, yellow side window, 45 degrees  | 1             | grabbed it on the edge                                                                         |
| 10    | Bottom right, yellow side window, 45 degrees | 0             | messed up position and pushed block out of view on first attempt, 3 attempts                   |
| 11    | Center, yellow side mirror, 90 degrees       | 1             |                                                                                                |
| 12    | Top left, yellow side mirror, 90 degrees     | 0             | position and angle messed up on first attempt, later attempts better gripper angle, 3 attempts |
| 13    | Top right, yellow side mirror, 90 degrees    | 0             | position and angle messed up on first try, later attempts pushed block, better than previous   |
| 14    | Bottom left, yellow side mirror, 90 degrees  | 0             | position wrong on all 3 tries — goes for middle most of the time                               |
| 15    | Bottom right, yellow side mirror, 90 degrees | 0             | again position wrong, goes for the middle and doesn't go to the bottom                         |
| 16    | Center, yellow side door, 135 degrees        | 1             | 2nd attempt worked, first attempt tipped block to the side                                     |
| 17    | Top left, yellow side door, 135 degrees      | 0             | generally goes for the middle position — possibly not enough edge cases in training            |
| 18    | Top right, yellow side door, 135 degrees     | 0             | touched the block but didn't manage to get position and angle right together                   |
| 19    | Bottom left, yellow side door, 135 degrees   | 1             | worked on 2nd attempt, 1st attempt aimed flatter (0°-like) and more central                    |
| 20    | Bottom right, yellow side door, 135 degrees  | 1             | block left a bit to the side in target plate                                                   |

## Result

- **Successes:** 9 / 20
- **Nominal success rate:** 45 %

## Observations (free-form, post-eval)

- **Position is the dominant failure axis:** Center 4/4, Bottom left 3/4, Top left 1/4, Bottom right 1/4, Top right 0/4. Orientation breakdown (0°: 3/5, 45°: 2/5, 90°: 1/5, 135°: 3/5) is secondary and partly explained by position effects below.
- **Hypothesized mechanisms by cell (to verify against recorded rollouts):**
  - Top left: arm struggles with full extension — reach-limit failures.
  - Top right: requires larger wrist/gripper rotation than center; position + angle rarely land together (0/4).
  - Bottom cells: success tracks whether part of the block sits close enough to the center of the workspace. 90° pushes the whole block away from center at both bottom cells → worst cell row; 135° leaves a graspable corner nearer center → bottom recovers (bottom left 135° success). Same logic: bottom left 0° succeeded via a central touch/tip; bottom right 0° could not.
  - Gripper angle per se seems NOT to be the driver at 90°: that wrist angle is close to the resting pose, i.e. kinematically easy — yet 90° is the worst orientation. Consistent with position/reach, not angle, doing the damage.
- **Common failure signature away from center:** policy drifts toward the middle of the workspace ("goes for the middle"), consistent with defaulting to the training distribution's mode when the block is far from it.
- Retry behavior present throughout (up to 3 attempts per 30 s budget) — closed-loop recovery exists but repeats similar errors in hard cells.
- **Next steps checks:** (1) training-data coverage by block position (and rotation) — is the mode central? (2) overlay commanded gripper trajectory vs. block position per failed cell to classify extension vs. rotation vs. go-to-middle failures.

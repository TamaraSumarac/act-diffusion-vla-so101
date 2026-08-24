# Diffusion Policy — Nominal Eval (20 trials)

- **Date:** 2026-08-22
- **Policy:** Diffusion, checkpoint 100000, DDIM-10 inference variant (`checkpoints_diffusion_baseline/100000_ddim10`)
- **Dataset trained on:** `TamaraSumarac/so101_policy_robustness` @ `e2dd884` (trimmed; sampler covered 20625 frames = 20975 − 50×7 preset drop_n_last_frames)
- **Deployment:** `lerobot-rollout`, strategy=episodic, inference=sync, device=mps, 30 fps; ~0.9 s chunk generation (think-pauses expected; see deployment note in results_diffusion.md)
- **Protocol:** frozen 5-position × 4-orientation grid, identical cells and order to eval_nominal_act.md; binary success = block at rest fully within plate rim; all trials scored, no discards; 30 s budget
- **Rollout dataset:** `TamaraSumarac/rollout_diffusion_nominal_eval` (local only)
- All blocks start with pink side of the block up

| Trial | Start cell                                   | Success (0/1) | Failure note (one phrase) |
|-------|----------------------------------------------|---------------|---------------------------|
| 1     | Center, yellow side window, 0 degrees        |               |                           |
| 2     | Top left, yellow side window, 0 degrees      |               |                           |
| 3     | Top right, yellow side window, 0 degrees     |               |                           |
| 4     | Bottom left, yellow side window, 0 degrees   |               |                           |
| 5     | Bottom right, yellow side window, 0 degrees  |               |                           |
| 6     | Center, yellow side window, 45 degrees       |               |                           |
| 7     | Top left, yellow side window, 45 degrees     |               |                           |
| 8     | Top right, yellow side window, 45 degrees    |               |                           |
| 9     | Bottom left, yellow side window, 45 degrees  |               |                           |
| 10    | Bottom right, yellow side window, 45 degrees |               |                           |
| 11    | Center, yellow side mirror, 90 degrees       |               |                           |
| 12    | Top left, yellow side mirror, 90 degrees     |               |                           |
| 13    | Top right, yellow side mirror, 90 degrees    |               |                           |
| 14    | Bottom left, yellow side mirror, 90 degrees  |               |                           |
| 15    | Bottom right, yellow side mirror, 90 degrees |               |                           |
| 16    | Center, yellow side door, 135 degrees        |               |                           |
| 17    | Top left, yellow side door, 135 degrees      |               |                           |
| 18    | Top right, yellow side door, 135 degrees     |               |                           |
| 19    | Bottom left, yellow side door, 135 degrees   |               |                           |
| 20    | Bottom right, yellow side door, 135 degrees  |               |                           |

## Result

- **Successes:** __ / 20
- **Nominal success rate:** __ %

## Observations (free-form, post-eval)

-

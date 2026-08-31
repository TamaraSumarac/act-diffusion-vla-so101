# SmolVLA Novel Color — Task String Change (20 trials)

- **Date:** 2026-08-31
- **Policy:** SmolVLA, checkpoint 100000 (`checkpoints_smolvla_baseline/100000/pretrained_model`)
- **Dataset trained on:** `TamaraSumarac/so101_policy_robustness` @ `e2dd884` (trimmed, 20975 frames)
- **Deployment:** `lerobot-rollout`, strategy=episodic, inference=sync, device=mps, 30 fps
- **Protocol:** fixed 5-position × 4-orientation grid (frozen for all policies); binary success = block at rest fully within plate rim; all trials scored, no discards; episode budget 30 s
- **Rollout dataset:** `TamaraSumarac/rollout_eval_novel_color_smolvla_string_change` (local only)
- - **Task string:** "Pick up the light blue block and place it in the plate" (changed from "pink block" — the only variable vs the novel_color grid block)
- **Episode mapping:** episode 0 = warm-up (void), scored trials = episodes 1–20
- **Pre-registered prediction:** success recovers toward nominal (~35%) → SmolVLA's novel-color collapse was instruction grounding; stays near novel_color grid level (~10%) → collapse is visual color brittleness independent of the string.
- **Block orientation:** light blue faces up, yellow faces toward window at 0° (matches novel_color grid block placement)

| Trial | Start cell                                   | Success (0/1) | Failure note (one phrase)                                                                                  |
|-------|----------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------|
| 1     | Center, yellow side window, 0 degrees        | 1             |                                                                                                            |
| 2     | Top left, yellow side window, 0 degrees      | 0             | tipped it on first try                                                                                     |
| 3     | Top right, yellow side window, 0 degrees     | 0             | almost picked it on first try but accidentally tipped it in region between starting region and target tray |
| 4     | Bottom left, yellow side window, 0 degrees   | 0             | tipped it sideways on first try                                                                            |
| 5     | Bottom right, yellow side window, 0 degrees  | 0             | almost succeded on 2nd try                                                                                 |
| 6     | Center, yellow side window, 45 degrees       | 1             |                                                                                                            |
| 7     | Top left, yellow side window, 45 degrees     | 0             | position was more close to middle                                                                          |
| 8     | Top right, yellow side window, 45 degrees    | 0             | tipped it and then was moving next to it                                                                   |
| 9     | Bottom left, yellow side window, 45 degrees  | 1             | tipped it on pickup                                                                                        |
| 10    | Bottom right, yellow side window, 45 degrees | 0             | was close but positionally sligtly off                                                                     |
| 11    | Center, yellow side mirror, 90 degrees       | 1             |                                                                                                            |
| 12    | Top left, yellow side mirror, 90 degrees     | 0             | going for the middle of the starting region close to block                                                 |
| 13    | Top right, yellow side mirror, 90 degrees    | 0             | same as previos one, but gripper did touch the block                                                       |
| 14    | Bottom left, yellow side mirror, 90 degrees  | 0             | going for the middle of the starting region close to block                                                 |
| 15    | Bottom right, yellow side mirror, 90 degrees | 0             | same as previous one                                                                                       |
| 16    | Center, yellow side door, 135 degrees        | 0             | wrong wrist angle position was good                                                                        |
| 17    | Top left, yellow side door, 135 degrees      | 0             | wrong wrist angle                                                                                          |
| 18    | Top right, yellow side door, 135 degrees     | 0             | positionally close but wrist angle was wrong                                                               |
| 19    | Bottom left, yellow side door, 135 degrees   | 0             | almost succeded, position slightly off so it moved the block around                                        |
| 20    | Bottom right, yellow side door, 135 degrees  | 0             | positionally close, wrist angle slightly wrong                                                             |

## Result

- **Successes:**  4 / 20
- **Nominal success rate:**  20%

## Observations 
- String task definitely affects the success rate: changing "pink block" to "light blue block" bumped success from 10% to 20%, and the arm no longer behaved as if it did not register the block — it localized and approached it like it does under nominal conditions. Remaining failures look like SmolVLA's usual ones (wrong wrist angle at 135°, seeking demo middle position, near misses), not like the "block not seen" failures from the original novel color run
- This result also suggests that string is not the only factor that matters for SmolVLA - it seems like visually SmolVLA does rely on color as well as we were not able to fully recover nominal result (35%).


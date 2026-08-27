# Perturbation Protocol — Robustness Grid

- Policies: ACT, SmolVLA. N=20 trials per condition per policy. All trials scored binary (object released at target in trial time).
- Trial order pre-generated. Policy blocks run sequentially within each condition (ACT then SmolVLA, back to back, ~20min each).
- Nominal reference: ACT 45%, SmolVLA 35% (20 trials each, standard start grid). Perturbations tested here will be read as deltas from this nominal.
- Start grid (all conditions except position shift): the standard 20-cell grid — 5 positions (center, top left, top right, bottom left, bottom right) × 4 orientations (0°, 45°, 90°, 135°).
- Warm-up: one throwaway SmolVLA rollout before each SmolVLA block (kernel compile ~7.6s must not land in a scored trial). This is important for block slide test, less for others, but keep it consistent.
- As some of these pertubrations can distrub hardware, we will do calibration gate before each hardware session: (e.g. pan≈26.7, lift≈−102.9, elbow≈97.4, wrist_flex≈63.6, wrist_roll≈0, gripper≈0.8).

---

## 1. Position shift 
- 4 fixed positions strictly outside the demo start region:
  1. Midway between start region and tray, along the Theraflu middle line.
  2. Midway between start region and tray, along the bottom of Theraflu edge.
  3. Directly below the start region, centered on it, on the follower-mount side.
  4. Directly below the tray, centered on it, on the follower-mount side.
- 4 trials per position = 16 total. Position fixed across its 4 trials, orientation varies (0°, 45°, 90°, 135°).
- Measures: whether each policy goes for the demo middle or actually adjusts to positions it has never seen.
- Hypothesis: ACT would seek middle more than SmolVLA.

## 2. Novel color
- Same cube pink side at the bottom, blue color at the top, purple on the side towards window at 0°.
- Standard 20-cell grid used in nominal tests.
- Hypothesis: SmolVLA might be more conditioned on the color than ACT is.

## 3. Distractor (pink sock)
- Standard pink block is the target.
- One clumped pink sock (tied to keep robust shape) placed exactly in between start region and tray 8cm away from table edge on camera side, not obstructing trajectory. 
- Standard 20-cell grid used in nominal tests for pink block, pink sock keeps same spot all 20 trials.
- Hypothesis: ACT ignores the sock (assuming it always goes for the middle); SmolVLA may get confused by wrong pink object. ACT success via vision-ignoring = memorization signature, not robustness.

## 4. Lighting
- Standard 20-cell grid used in nominal tests with pink block. 
- One repeatable change: desk lamp ON and kitchen lights on. 
- Hypothesis: both policies get worse, but SmolVLA is more robust to this change.

## 5. Novel object (pink sock as target object)
- Clumped pink sock replaces cube as target object.
- Standard 20-cell grid used in nominal tests. 
- This test along with novel color can tell us how well do policies adjust to change in color of the object or shape of the object.
- Scoring: if it goes for the sock but the gripper can't hold it (slips out because it's not a block), that is still a fail (0) — but write "correct reach, sock slipped" in the failure note. This way the analysis can separate "policy never found the sock" from "policy found it, gripper physics failed."
- Hypothesis: SmolVLA can better adjust to new object than ACT given its pre-trained prior.

## 6. Block slide
- Standard 20-cell grid used in nominal tests with pink block. 
- Mid-rollout slide: ~4cm, hand pushed, destination must remain inside demo region.
- Triggered by event, not stopwatch — slide when gripper reaches right edge of the Theraflu starting region. Same for both policies.
- Direction of shift per trial:

| Trial | Start cell                                   | Shift                                            | Success (0/1) | Failure note (one phrase) |
|-------|----------------------------------------------|--------------------------------------------------|---------------|---------------------------|
| 1     | Center, yellow side window, 0 degrees        | shift 4cm to the bottom edge                     |               |                           |
| 2     | Top left, yellow side window, 0 degrees      | shift 4cm to the bottom edge                     |               |                           |
| 3     | Top right, yellow side window, 0 degrees     | shift 4cm to the bottom edge                     |               |                           |
| 4     | Bottom left, yellow side window, 0 degrees   | shift 4cm to the top edge (more middle position) |               |                           |
| 5     | Bottom right, yellow side window, 0 degrees  | shift 4cm to the top edge (more middle position) |               |                           |
| 6     | Center, yellow side window, 45 degrees       | shift 4cm to the bottom edge                     |               |                           |
| 7     | Top left, yellow side window, 45 degrees     | shift 4cm to the bottom edge                     |               |                           |
| 8     | Top right, yellow side window, 45 degrees    | shift 4cm to the bottom edge                     |               |                           |
| 9     | Bottom left, yellow side window, 45 degrees  | shift 4cm to the top edge (more middle position) |               |                           |
| 10    | Bottom right, yellow side window, 45 degrees | shift 4cm to the top edge (more middle position) |               |                           |
| 11    | Center, yellow side mirror, 90 degrees       | shift 4cm to the bottom edge                     |               |                           |
| 12    | Top left, yellow side mirror, 90 degrees     | shift 4cm to the bottom edge                     |               |                           |
| 13    | Top right, yellow side mirror, 90 degrees    | shift 4cm to the bottom edge                     |               |                           |
| 14    | Bottom left, yellow side mirror, 90 degrees  | shift 4cm to the top edge (more middle position) |               |                           |
| 15    | Bottom right, yellow side mirror, 90 degrees | shift 4cm to the top edge (more middle position) |               |                           |
| 16    | Center, yellow side door, 135 degrees        | shift 4cm to the bottom edge                     |               |                           |
| 17    | Top left, yellow side door, 135 degrees      | shift 4cm to the bottom edge                     |               |                           |
| 18    | Top right, yellow side door, 135 degrees     | shift 4cm to the bottom edge                     |               |                           |
| 19    | Bottom left, yellow side door, 135 degrees   | shift 4cm to the top edge (more middle position) |               |                           |
| 20    | Bottom right, yellow side door, 135 degrees  | shift 4cm to the top edge (more middle position) |               |                           |

Caveats:
- All perturbations are moving block along y axis and toward "easier to grab" location. This is chosen so that failure is not blamed on position is hard to reach.
- Want to test here if policy adjusts to change at all. Need to make sure to take latency into account as ACT replans ~351ms, SmolVLA ~1s - so if adjustment would happen it would happen slower for SmolVLA. So focus here is whether adjustment happened not so much on how fast it happened.
- Hypothesis: ACT would do more autopilot - going for the middle of demo mode, SmolVLA would be better at adjusting to new position.

---

## Abort criteria (any block)
Audible grinding, visible sawtooth oscillation, or calibration test failed → stop, recalibrate and check each motor is giving the same measurement, resume.

## Trial log schema
trial_id, policy, condition, start cell, slide direction (cond. 6 only), success 0/1, failure note, eval dataset repo-id, episode index within video

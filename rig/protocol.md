# Phase 2 Recording Protocol — frozen [2026-08-14]

All 50 dataset episodes and all Week 9 nominal-condition trials follow this document.
Any deviation = not nominal. Week 9 perturbations are defined relative to this file.

## Task string (verbatim, used at record time)
"Pick up the pink block and place it in the colorful plate"

## Object
- Pink/purple foam buffer block (nail file), 9.1 × 3.6 × 2.7 cm
- Grasped across dimension: 3.6 cm (confirm comfortable gripper close)
- Face up: pink side
- Start placement: Object always placed with pink side pointing up; position and rotation of the object 
relative to the start region are randomized (any angle, continuous), block fully on the sheet.


## Start region
- Blue cardboard sheet (Theraflu packaging), blue branded side up
- Sheet dimensions: 18.2 × 13 cm
- Taped flat at all four corners; fold seams flattened
- Position: 9cm cm from camera side of the table (x-axis in camera plane),
            43 cm from follower side of the table (y-axis in camera plane)
- Definition: IN-distribution = block fully on sheet. OOD (Week 9) = block off sheet.
- Known fact: sheet is a learnable visual landmark; Week 9 off-sheet perturbation
  entangles position generalization with landmark dependence (acknowledged by design).
- Known visual: no significant glare or sheen on sheet or plate under nominal lighting. 
The target plate's corner closest to camera is partially chopped in the camera frame: a triangular 
area of ~4×4 cm of the plate surface is not visible. Verified that a block placed in this area 
still appears clearly in the frame.

## Target
- Patterned ceramic plate (blue/cream/orange), 18 × 15 cm, rim height ~1.6 cm
- Position: 11 cm from camera side of the table (x-axis in camera plane), 
            7.5 cm from follower side of the table (y-axis in camera plane) 
            taped to the table at the bottom
- Distance between start sheet center and plate center: 31 cm
- Success criterion (binary): block at rest fully within the plate rim,
  no part touching the table, at episode end. Anything else = failure.

## Camera
- Index 0, OpenCV, 640×480 @ 30 fps
- Mount: 17 cm above table surface, 13 cm horizontal from start-sheet center,
  ~40° angular tilt relative to vertical, 1/2 front view (50% side 50% front)
- Mount footprint tape-marked: yes
- Reference photos in repo: camera_setup.jpg (side view of rig), camera_view.png (feed)
- Verified in feed: block visible everywhere on sheet, plate visible,
  gripper in frame through full pick–carry–drop–retract.
- Excluded from frame: leader arm, operator hands

## Lighting
- Overhead light directly above setup: OFF
- Curtains: fully closed
- Other living room lamps: on, 100% brightness
- Kitchen lights: off
- Nominal time-of-day: noon

## Home poses & reset ritual
- Follower home: gripper closed, arm settled in gravity resting pose, 
reached by holding leader at its rest position (see camera_setup.jpg); 
positioned clear of start/target zones. Every episode starts from this pose.
- Leader rest: gripper closed, arm settled in gravity resting pose (see camera_setup.jpg)
- Reset sequence (every episode):
  1. Follower to home via leader
  2. Leader to rest position
  3. Block placed by hand: random position fully on sheet, random angle, pink up
  4. Hands out of frame
  5. Record
- Reset target time: ~15 s

## Episode definition
- Starts: follower at home
- Ends: block released in plate + gripper retracted clear of plate
- Target duration: ~20 s
- Grasp strategy (one, all 50 episodes): top-down, gripper across
  3.6 cm dimension
- Redo criteria — discard and immediately re-record if ANY of:
  hesitation / stall mid-motion, missed first grasp, multi-attempt grasp,
  mid-motion correction, block dropped, block clipped plate rim on entry
- Redo mechanism: re-record key from lerobot-record, filled in after dry run

## Hardware config (ratified)
- Follower: /dev/tty.usbmodem5B610326031, id so101_follower
- Leader:   /dev/tty.usbmodem5B610326051, id sso101_leader
- Hub: robot hardware (both arms + camera) on Acer 10Gbps hub on USB-C port 2;
  display dock on port 1; never mixed
- Env: conda `lerobot`, LeRobot v0.6.1, Python 3.12.13, Terminal.app only

## Record command (verified [date], pasted from dry run)
- Keyboard controls confirmed: end episode early = n,
  re-record current episode = r, stop/save = q
- Local dataset cache path: ~/.cache/huggingface/lerobot/TamaraSumarac/

## Dataset target
- 50 clean episodes, single task string, pushed to HF Hub as [TamaraSumarac]/[so101_policy_robustness]
- Also added local root: --dataset.root=/Users/tamara/datasets/so101_policy_robustness \
- This dataset is FROZEN once pushed: all of Weeks 7–9 and Phase 3 consume it unchanged

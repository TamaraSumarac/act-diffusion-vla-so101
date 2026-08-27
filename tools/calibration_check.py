#!/usr/bin/env python
"""Calibration gate: verify the follower against the known-goods readout.

Procedure: place the arm in the reference pose (same pose used when the
known-goods numbers were recorded), then:

    python rig/calibration.py

PASS  -> the raw-count -> degrees dictionary is intact; evals are interpretable.
FAIL  -> stop. Recalibrate, re-verify, only then run trials. A silent one-tooth
         horn slip (motor 5) shows up here as a wrist_roll offset.

Run: start of each hardware session, after the slide block, after any jam or
hard contact.
"""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

PORT = "/dev/tty.usbmodem5B610326031"
ROBOT_ID = "so101_follower"

# Known-goods reference readout: {joint: (expected, tolerance)}.
# Tolerances are per-joint: pan and wrist_flex depend on hand-placement of the
# reference pose (loose, +/-10), the rest are mechanically determinate (+/-3).
# Any real horn slip is >= one spline tooth (~14.4 deg) and exceeds both.
KNOWN_GOODS = {
    "shoulder_pan.pos":  (26.7,   10.0),
    "shoulder_lift.pos": (-102.9,  3.0),
    "elbow_flex.pos":    (97.4,    3.0),
    "wrist_flex.pos":    (63.6,   10.0),
    "wrist_roll.pos":    (0.0,     3.0),
    "gripper.pos":       (0.8,     3.0),
}

def main():
    cfg = SO101FollowerConfig(port=PORT, id=ROBOT_ID)
    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)  # never re-calibrate here; we're checking, not writing
    try:
        obs = robot.get_observation()
    finally:
        robot.disconnect()

    missing = [k for k in KNOWN_GOODS if k not in obs]
    if missing:
        print("ERROR: expected joint keys not in observation:", missing)
        print("Observation keys seen:", sorted(k for k in obs if k.endswith(".pos")))
        raise SystemExit(2)

    print(f"{'joint':<20}{'expected':>10}{'measured':>10}{'delta':>8}{'tol':>6}  verdict")
    failed = []
    for joint, (expected, tol) in KNOWN_GOODS.items():
        measured = float(obs[joint])
        delta = measured - expected
        ok = abs(delta) <= tol
        if not ok:
            failed.append(joint)
        print(f"{joint:<20}{expected:>10.1f}{measured:>10.1f}{delta:>+8.1f}{tol:>6.1f}  {'ok' if ok else 'FAIL'}")

    if failed:
        print(f"\nFAIL: {', '.join(failed)} outside per-joint tolerance.")
        print("Do NOT run trials. Recalibrate, re-verify against known-goods, then rerun this gate.")
        raise SystemExit(1)
    print("\nPASS: all joints within per-joint tolerance of known-goods.")

if __name__ == "__main__":
    main()
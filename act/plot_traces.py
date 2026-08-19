"""Overlay commanded action vs observed state per joint for one episode of a
rollout dataset. Verdict tool for 'policy error vs tracking error'.

Usage:
    python plot_traces.py <episode_index>            # e.g. 13 for trial 14
"""
import sys

import matplotlib.pyplot as plt
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "TamaraSumarac/rollout_act_nominal_eval_20260818_104621"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

def main(ep):
    ds = LeRobotDataset(REPO_ID, episodes=[ep])
    actions = torch.stack([ds[i]["action"] for i in range(len(ds))]).numpy()
    states = torch.stack([ds[i]["observation.state"] for i in range(len(ds))]).numpy()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    t = range(len(ds))
    for j, (name, ax) in enumerate(zip(JOINTS, axes.flat)):
        ax.plot(t, actions[:, j], label="action (commanded)", lw=1.5)
        ax.plot(t, states[:, j], label="state (actual)", lw=1.5, alpha=0.8)
        gap = abs(actions[:, j] - states[:, j]).mean()
        ax.set_title(f"{name}   mean |cmd-state| = {gap:.2f} deg")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend()
    axes.flat[-1].set_xlabel("frame (30 fps)")
    fig.suptitle(f"Episode {ep} (trial {ep+1}) — commanded vs actual, per joint")
    fig.tight_layout()
    out = f"traces_ep{ep}.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    plt.show()

if __name__ == "__main__":
    main(int(sys.argv[1]))

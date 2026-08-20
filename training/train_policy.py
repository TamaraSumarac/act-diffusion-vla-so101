"""Policy training (ACT / Diffusion / SmolVLA) with per-episode head/tail trims applied at the sampler level.

Stock lerobot_train, one interception: EpisodeAwareSampler receives per-episode
from/to frame-index arrays; we shift them inward using trim_frames.json
(head/tail detected from joint-state motion; 15-frame safety margin) so idle
prefix/tail frames are never sampled. Dataset, stats, and indexing untouched.
"""
import json

import numpy as np

import lerobot.scripts.lerobot_train as lerobot_train

TRIM_PATH = "/home/ubuntu/trim_frames.json"
MARGIN = 15

_OrigSampler = lerobot_train.EpisodeAwareSampler


class TrimmedEpisodeAwareSampler(_OrigSampler):
    def __init__(self, from_indices, to_indices, *args, **kwargs):
        with open(TRIM_PATH) as f:
            trims = {int(k): v for k, v in json.load(f).items()}
        from_indices = np.asarray(from_indices, dtype=np.int64).copy()
        to_indices = np.asarray(to_indices, dtype=np.int64).copy()
        for ep, t in trims.items():
            from_indices[ep] += max(0, t["head"] - MARGIN)
            to_indices[ep] -= max(0, t["tail"] - MARGIN)
        assert (to_indices > from_indices).all(), "trim produced an empty episode"
        super().__init__(from_indices, to_indices, *args, **kwargs)
        print(f"[trim] sampler covers {len(self)} frames after per-episode trims")


lerobot_train.EpisodeAwareSampler = TrimmedEpisodeAwareSampler

if __name__ == "__main__":
    lerobot_train.main()

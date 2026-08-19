import json
import torch

class TrimmedDataset(torch.utils.data.Dataset):
    """Drops pre-head and idle-tail frames per episode.
    trim_frames.json stores raw head/tail counts; margin applied here."""

    def __init__(self, dataset, trim_path="/home/ubuntu/trim_frames.json", margin=15):
        self._ds = dataset
        with open(trim_path) as f:
            trims = {int(k): v for k, v in json.load(f).items()}
        self.indices = []
        for ep in range(dataset.num_episodes):
            e = dataset.meta.episodes[ep]
            start = int(e["dataset_from_index"])
            end = int(e["dataset_to_index"])   # exclusive
            t = trims.get(ep, {"head": 0, "tail": 0})
            lo = max(0, t["head"] - margin)
            hi = (end - start) - max(0, t["tail"] - margin)
            self.indices.extend(range(start + lo, start + hi))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self._ds[self.indices[i]]

    def __getattr__(self, name):
        return getattr(self._ds, name)

"""Off-robot smoke test: load a trained policy checkpoint WITH its pre/post
processor pipelines (mirroring lerobot_eval.py's loading path) and run one
dataset frame through the full stack:

    raw frame -> preprocessor (device, normalize) -> policy -> postprocessor (unnormalize)

Pass = predicted first action lands within a few degrees of the ground-truth
action for that frame.

Note: checkpoints trained on CUDA save 'cuda' inside their processor configs;
on Apple silicon we override the device step to the policy's actual device
(mps). Same mechanism lerobot_eval.py uses via preprocessor_overrides.

Usage:
    python smoke_test_policy.py checkpoints_act_baseline/100000/pretrained_model
"""
import sys

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy

REPO_ID = "TamaraSumarac/so101_policy_robustness"
FRAME_IDX = 100

def main(ckpt_path):
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.eval()
    device = next(policy.parameters()).device
    print("policy device:", device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("processors loaded OK")

    ds = LeRobotDataset(REPO_ID)
    frame = ds[FRAME_IDX]
    batch = {
        k: v.unsqueeze(0).to(device)
        for k, v in frame.items()
        if isinstance(v, torch.Tensor)
    }

    batch = preprocessor(batch)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    action = postprocessor(chunk[:, 0])          # first action of the chunk, unnormalized

    print("chunk shape:", tuple(chunk.shape))     # expect (1, 100, 6)
    print("predicted (deg):", [round(x, 2) for x in action.squeeze().cpu().tolist()])
    print("truth     (deg):", [round(x, 2) for x in frame["action"].tolist()])

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints_act_baseline/100000/pretrained_model")
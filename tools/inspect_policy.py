"""Off-robot architecture inspection, policy-agnostic: load a checkpoint and
print its hyperparameters, module tree, and parameter counts. No dataset,
no robot, no inference.

Usage:
    python tools/inspect_policy.py checkpoints_act_baseline/100000/pretrained_model
    python tools/inspect_policy.py checkpoints_act_baseline/100000/pretrained_model --tree
"""
import sys

from lerobot.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main(ckpt_path, show_tree=False):
    cfg = PreTrainedConfig.from_pretrained(ckpt_path)
    policy = get_policy_class(cfg.type).from_pretrained(ckpt_path).eval()

    print(f"=== {cfg.type} hyperparameters ===")
    for k, v in sorted(vars(cfg).items()):
        if not k.startswith("_"):
            print(f"  {k:32s} {v}")

    print(f"\n=== parameters ===")
    print(f"  {'TOTAL':32s} {n_params(policy)/1e6:8.2f} M")
    # top-level children of the inner model (backbone / vae_encoder / encoder / decoder / ...)
    inner = getattr(policy, "model", policy)
    for name, m in inner.named_children():
        print(f"  {name:32s} {n_params(m)/1e6:8.2f} M")

    if show_tree:
        print(f"\n=== module tree ===")
        print(policy)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], show_tree="--tree" in sys.argv)

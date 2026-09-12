"""Off-robot smoke test, policy-agnostic: load any trained checkpoint WITH its
pre/post processor pipelines, run one dataset frame through the full stack,
and time the inference call.

    raw frame -> preprocessor (device, normalize) -> policy -> postprocessor -> degrees

Pass = predicted first action within a few degrees of ground truth, and
per-chunk inference time compatible with the 30 Hz control budget.

Usage:
    python tools/smoke_test_policy.py checkpoints_act_baseline/100000/pretrained_model
    python tools/smoke_test_policy.py checkpoints_diffusion_baseline/100000/pretrained_model
    python tools/smoke_test_policy.py checkpoints_smolvla_baseline/100000/pretrained_model
"""
import sys
import time

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

REPO_ID = "TamaraSumarac/so101_policy_robustness"
FRAME_IDX = 100
TASK = "Pick up the pink block and place it in the plate"   # consumed by VLA policies only

def main(ckpt_path):
    cfg = PreTrainedConfig.from_pretrained(ckpt_path)
    policy_cls = get_policy_class(cfg.type)
    kwargs = {}
    if cfg.type == "diffusion":
        kwargs = {"noise_scheduler_type": "DDIM", "num_inference_steps": 10}
    policy = policy_cls.from_pretrained(ckpt_path, **kwargs)
    if cfg.type == "diffusion":
        print("scheduler check:", policy.config.noise_scheduler_type,
              policy.config.num_inference_steps,
              type(policy.diffusion.noise_scheduler).__name__)
    policy.eval()
    # Deployment override for diffusion: DDIM sampling with few steps (the
    # paper's inference recipe). No effect on other policy types.
    if cfg.type == "diffusion":
        policy.config.noise_scheduler_type = "DDIM"
        policy.config.num_inference_steps = 10
    device = next(policy.parameters()).device
    print(f"policy type: {cfg.type}   device: {device}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("processors loaded OK")

    fps = 30
    delta_timestamps = {}
    if getattr(policy.config, "observation_delta_indices", None):
        for key in ["observation.images.front", "observation.state"]:
            delta_timestamps[key] = [i / fps for i in policy.config.observation_delta_indices]
    if getattr(policy.config, "action_delta_indices", None):
        delta_timestamps["action"] = [i / fps for i in policy.config.action_delta_indices]
    ds = LeRobotDataset(REPO_ID, delta_timestamps=delta_timestamps or None)
    frame = ds[FRAME_IDX]
    batch = {k: v.unsqueeze(0).to(device) for k, v in frame.items() if isinstance(v, torch.Tensor)}
    batch["task"] = [TASK]   # ignored by non-VLA policies

    batch = preprocessor(batch)

    if "--graph" in sys.argv:
        from torchview import draw_graph
        model = policy.model
        b = dict(batch)
        if cfg.type == "act":
            b["observation.images"] = [b[k] for k in policy.config.image_features]
            model.train()          # dataset frame has "action" -> VAE branch is drawn too
        g = draw_graph(model, input_data={"batch": b}, depth=1,
                        expand_nested=True, save_graph=True,
                        filename=f"{cfg.type}_graph", directory=".")
        model.eval()
        print("graph saved ->", f"{cfg.type}_graph.png")
        # return

    # with torch.no_grad(): #
    #     t0 = time.perf_counter()
    #     chunk = policy.predict_action_chunk(batch)
    #     t1 = time.perf_counter()

    with torch.no_grad():
        policy.predict_action_chunk(batch)          # warm-up, discard
        if device.type == "mps": torch.mps.synchronize()
        t0 = time.perf_counter()
        chunk = policy.predict_action_chunk(batch)
        if device.type == "mps": torch.mps.synchronize()
        t1 = time.perf_counter()

    action = postprocessor(chunk[:, 0])
    n_actions = chunk.shape[1]
    dt = t1 - t0
    print("chunk shape:", tuple(chunk.shape))
    print("predicted (deg):", [round(x, 2) for x in action.squeeze().cpu().tolist()])
    gt = frame["action"]
    gt0 = gt[0] if gt.dim() > 1 else gt
    print("truth     (deg):", [round(x, 2) for x in gt0.tolist()])
    print(f"inference: {dt*1000:.0f} ms per chunk of {n_actions} actions "
          f"-> budget {n_actions/30:.2f} s at 30 Hz -> margin {'OK' if dt < n_actions/30 else 'TOO SLOW'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
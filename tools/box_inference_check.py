# gate0_gpu_inference_bench.py
#
# Provenance: written 2026-08-31 (Week 9, SmolVLA latency investigation).
# Purpose: Gate 0 of the remote-inference experiment — measure bare GPU
#   chunk-inference time for the baseline SmolVLA checkpoint on a cloud GPU
#   (Lambda A10), with no robot and no network in the loop.
#   Compare against ~1s/chunk observed on MacBook Pro (MPS) during Phase 2.
#   Decision rule: ~0.15s median -> proceed to Gate 1 (network payload RTT);
#   >=0.5s median -> GPU win is marginal once network cost is added.
#
# Usage (on the box, venv active):
#   python gate0_gpu_inference_bench.py --ckpt /home/ubuntu/checkpoints_smolvla_baseline_100k
#
# Step 1 prints the checkpoint's input_features so you can confirm the
# dummy observation below matches (camera keys, resolution, state dim).
# Adjust CAMERA_KEYS / RESOLUTION / STATE_DIM / TASK if they differ.

import argparse
import json
import os
import time

import torch

# ---- rig-specific settings (verify against printed input_features) ----
CAMERA_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]
RESOLUTION = (256, 256)
STATE_DIM = 6
TASK = "Pick up the pink block and place it in the plate"

N_WARMUP = 3   # kernel compile / autotune land here (cf. episode-0 convention)
N_TIMED = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to pretrained_model dir")
    args = ap.parse_args()

    # Step 1: print input_features from config for verification
    cfg_path = os.path.join(args.ckpt, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    print("=== input_features from config.json ===")
    print(json.dumps(cfg.get("input_features"), indent=2))
    print("=======================================\n")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    assert torch.cuda.is_available(), "CUDA not available — wrong box/env?"
    p = SmolVLAPolicy.from_pretrained(args.ckpt).to("cuda").eval()

    h, w = RESOLUTION
    obs = {"observation.state": torch.zeros(1, STATE_DIM, device="cuda")}
    for k in CAMERA_KEYS:
        obs[k] = torch.rand(1, 3, h, w, device="cuda")

    # Tokenize the task the way the rollout preprocessor does
    # (policy expects observation.language.tokens / attention_mask, not a raw string)
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(p.config.vlm_model_name).tokenizer
    enc = tok(
        TASK + "\n",                       # SmolVLA convention: newline-terminated task
        padding="max_length",
        max_length=p.config.tokenizer_max_length,
        return_tensors="pt",
    )
    obs["observation.language.tokens"] = enc["input_ids"].to("cuda")
    obs["observation.language.attention_mask"] = enc["attention_mask"].to("cuda").bool()

    with torch.no_grad():
        for _ in range(N_WARMUP):
            p.select_action(obs)
            p.reset()  # force a fresh chunk (don't replay the action queue)

        ts = []
        for _ in range(N_TIMED):
            p.reset()
            t0 = time.time()
            p.select_action(obs)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)

    ts.sort()
    print(f"median chunk inference: {ts[len(ts) // 2]:.4f} s")
    print("all sorted:", [round(t, 3) for t in ts])
    print(f"\nMac MPS reference: ~1 s/chunk. Speedup: ~{1.0 / ts[len(ts) // 2]:.1f}x")


if __name__ == "__main__":
    main()

# side_by_side_gif.py
#
# Purpose: render the same episode from two rollout datasets as one
#   side-by-side GIF (left = A, right = B), for qualitative comparison of
#   policy behavior on an identical start cell (e.g. SmolVLA v1 vs v2).
#
# Usage (repo root, lerobot env):
#   python tools/side_by_side_gif.py \
#     --repo_a TamaraSumarac/rollout_smolvla_nominal_eval_20260824_084125 \
#     --repo_b TamaraSumarac/rollout_smolvla_v2_ep66_nominal_eval_20260901_075555 \
#     --episode 18 \
#     --label_a "SmolVLA 50 ep" --label_b "SmolVLA 66 ep" \
#     --out perturbation/follow-up\ results/trial19_v1_vs_v2.gif
#
# Notes:
# - Reads frames through LeRobotDataset (handles the chunked video format),
#   so it works for local-only or Hub datasets alike.
# - Episodes may differ in length; the shorter one holds its last frame.
# - Output is downsampled (fps, width) to keep GIF size sane for a README.

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERA_KEY = "observation.images.front"


def load_frames(repo_id: str, episode: int, width: int) -> list[Image.Image]:
    ds = LeRobotDataset(repo_id, episodes=[episode])
    frames = []
    for i in range(len(ds)):
        img = ds[i][CAMERA_KEY]  # (C, H, W) float in [0, 1]
        if isinstance(img, torch.Tensor):
            img = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        h = int(pil.height * width / pil.width)
        frames.append(pil.resize((width, h), Image.BILINEAR))
    return frames


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_a", required=True)
    ap.add_argument("--repo_b", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=320, help="per-panel width in px")
    ap.add_argument("--src_fps", type=int, default=30)
    ap.add_argument("--gif_fps", type=int, default=10)
    args = ap.parse_args()

    fa = load_frames(args.repo_a, args.episode, args.width)
    fb = load_frames(args.repo_b, args.episode, args.width)
    print(f"A: {len(fa)} frames | B: {len(fb)} frames")

    n = max(len(fa), len(fb))
    step = max(1, args.src_fps // args.gif_fps)
    out_frames = []
    for i in range(0, n, step):
        a = label(fa[min(i, len(fa) - 1)], args.label_a)
        b = label(fb[min(i, len(fb) - 1)], args.label_b)
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width + b.width + 4, h), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 4, 0))
        out_frames.append(canvas)

    out_frames[0].save(
        args.out,
        save_all=True,
        append_images=out_frames[1:],
        duration=int(1000 / args.gif_fps),
        loop=0,
        optimize=True,
    )
    print(f"wrote {args.out} ({len(out_frames)} frames @ {args.gif_fps} fps)")


if __name__ == "__main__":
    main()

# Cloud training runbook (Lambda, single A10)

The repo remembers; the box forgets. Per-run flow — box is disposable, everything
of value is scp'd back or on wandb before terminating.

## 1. Launch + connect
- Lambda console → single A10 (24 GB) → no filesystem attached → launch.
- `ssh ubuntu@<IP>`
- **`tmux new -s train`** immediately — everything runs inside tmux
  (reconnect after drops: `tmux attach -t train`; detach: Ctrl-b d).

## 2. Ship files from the Mac (second terminal tab)
    scp "/Users/tamara/Google Drive/act-diffusion-vla-so101/trim/trim_frames_v2.json" \
        "/Users/tamara/Google Drive/act-diffusion-vla-so101/training/train_policy.py" \
        "/Users/tamara/Google Drive/act-diffusion-vla-so101/training/setup_box.sh" \
        "/Users/tamara/Google Drive/act-diffusion-vla-so101/training/launch_smolvla_v2.sh" \
        ubuntu@<IP>:~/
    # (v1 runs used trim/trim_frames.json; TRIM_PATH in train_policy.py must match the file shipped)

## 3. On the box (inside tmux)
    bash setup_box.sh          # python 3.12 + venv + lerobot (see script)
    source ~/venv/bin/activate
    pip install "lerobot[smolvla]==0.6.1"
    hf auth login              # HF token (read scope)
    wandb login                # current API key
    python -c "import torch; print(torch.cuda.is_available())"   # must print True
    bash launch_smolvla_v2.sh

## 4. If losses go to Nan run reference code:
    python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/pusht_probe --job_name=pusht_probe \
    --batch_size=8 --steps=1000 --wandb.enable=false

## 5. Watchlist, first minutes
- `[trim] sampler covers 28180 frames` — must match (v1 was 20975); wrong number = stop.
- Loss falling, step time stable, checkpoints landing in outputs/.
- New ImportError naming an extra → `pip install "lerobot[<extra>]==0.6.1"`, relaunch.

## 6. After the run
- Pull checkpoints to the Mac: `scp -r ubuntu@<IP>:~/outputs/<run>/checkpoints/<step> <repo>/checkpoints_<run>/`
- Terminate the instance (not stop). Check Lambda Storage tab is empty.
- Known-good runs: ACT ~2.5 h at 11 steps/s on A10; loss scales differ per policy
  (ACT: L1+KL; Diffusion: denoising MSE) — compare shapes, not values.

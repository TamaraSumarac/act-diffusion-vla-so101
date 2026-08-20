#!/bin/bash
# One-time environment setup on a fresh Lambda box (Ubuntu, may ship python 3.10).
# Installs python 3.12, creates ~/venv, installs pinned lerobot with extras.
set -e

if ! command -v python3.12 >/dev/null; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt update
  sudo apt install -y python3.12 python3.12-venv
fi

python3.12 -m venv ~/venv
source ~/venv/bin/activate
pip install "lerobot[dataset,training]==0.6.1" wandb
echo "setup done — run: source ~/venv/bin/activate"

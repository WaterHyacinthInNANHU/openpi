#!/usr/bin/env bash
# pi05-DROID LoRA finetune on RLinf FR3 bench data (franka/test_finetune).
# Prereq:
#   1) dataset fully transferred to /data1/Franka_RealRobot/test-finetune
#   2) runme_preflight_franka.sh passed (norm-stats fit + format validation)
# Base ckpt: gs://openpi-assets/checkpoints/pi05_droid/params (auto-downloaded
# to ~/.cache/openpi on first run, ~10+ GB).
# Checkpoints land in /data1/Franka_RealRobot/checkpoints/pi05_droid_franka_lora/<exp>.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
export HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home

cd /data1/Franka_RealRobot/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_droid_franka_lora \
  --exp-name="${EXP_NAME:-pi05_droid_franka_lora_v0}" \
  --overwrite

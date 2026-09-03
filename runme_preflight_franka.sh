#!/usr/bin/env bash
# Preflight for pi05_droid_franka_lora — run AFTER the dataset transfer finishes.
#   1) DROID norm-stats fit check (our normalized-vel actions vs DROID rad/s stats)
#   2) RLinf's end-to-end format validation (key mapping + DroidInputs build)
set -euo pipefail

export HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home
cd /data1/Franka_RealRobot/openpi

uv run scripts/franka_preflight_norm_check.py

uv run python /data1/Franka_RealRobot/RLinf/examples/embodiment/validate_droid_lerobot.py \
  --repo-id franka/test_finetune

echo "PREFLIGHT DONE — read the VERDICT above before launching runme_finetune_franka.sh"

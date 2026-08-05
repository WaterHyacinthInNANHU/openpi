#!/usr/bin/env bash
# FALLBACK ONLY — compute our own norm stats for pi05_droid_franka_lora.
# Use this only if runme_preflight_franka.sh says the DROID stats don't fit.
# After it finishes, edit the pi05_droid_franka_lora config in
# src/openpi/training/config.py: drop the gs:// AssetsConfig override so
# training (and later serving) load the locally computed stats instead.
set -euo pipefail

export HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home
cd /data1/Franka_RealRobot/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_droid_franka_lora

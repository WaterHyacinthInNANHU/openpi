#!/usr/bin/env bash
# Launch Calibra50 training (pi05_droid_franka_lora_5k_calibra50) on all 8
# labserver GPUs. Dataset: franka/test_finetune_idlefiltered_calibra50
# (8 episodes / 2121 frames, Calibra keep-0.5 coreset). All settings
# identical to pi05_droid_franka_lora_5k (v2) except the dataset.
# Runs detached (invoke with: setsid bash ... > /tmp/train_v2_calibra50.log 2>&1 < /dev/null &)
set -u
cd /data1/Franka_RealRobot/openpi || exit 1

export HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home
export OPENPI_DATA_HOME=/data1/vla-reasoning/openpi-cache
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_MODE=offline
UVPY=$HOME/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=/data1/Franka_RealRobot/openpi/.venv/lib/python3.11/site-packages:\
/data1/Franka_RealRobot/openpi/src:/data1/Franka_RealRobot/openpi/packages/openpi-client/src

mkdir -p /data1/vla-reasoning/openpi-cache
exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$UVPY" scripts/train.py pi05_droid_franka_lora_5k_calibra50 \
  --exp-name=pi05_droid_franka_lora_5k_calibra50_v0 --overwrite

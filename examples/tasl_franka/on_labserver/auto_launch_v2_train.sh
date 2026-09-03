#!/usr/bin/env bash
# Auto-launch v2 training (pi05_droid_franka_lora_5k) once all 8 GPUs are free.
# Polls gpuq every 5 min. When >=7 GPUs are free, stops vla-reasoning's own
# serve.py (the 592MiB process pinning GPU 0) and launches training on all 8.
set -u

LOG=/tmp/auto_launch_v2.log
TRAIN_LOG=/tmp/train_v2.log
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

while true; do
  FREE=$(~/.local/bin/gpuq --json 2>/dev/null | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(len(d.get('free_gpus',[])))")
  echo "$(date '+%F %T') free_gpus=$FREE" >> "$LOG"
  if [ "${FREE:-0}" -ge 7 ]; then
    echo "$(date '+%F %T') freeing our own serve.py (GPU-0 holder)" >> "$LOG"
    pkill -f "scripts/serve.py --run-dir" 2>/dev/null
    sleep 5
    FREE=$(~/.local/bin/gpuq --json 2>/dev/null | python3 -c \
      "import json,sys; d=json.load(sys.stdin); print(len(d.get('free_gpus',[])))")
    echo "$(date '+%F %T') recheck free_gpus=$FREE" >> "$LOG"
    if [ "$FREE" = "8" ]; then
      echo "$(date '+%F %T') launching pi05_droid_franka_lora_5k" >> "$LOG"
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$UVPY" scripts/train.py \
        pi05_droid_franka_lora_5k --exp-name=pi05_droid_franka_lora_5k_v0 \
        --overwrite >> "$TRAIN_LOG" 2>&1
      echo "$(date '+%F %T') training exited rc=$?" >> "$LOG"
      break
    fi
  fi
  sleep 300
done

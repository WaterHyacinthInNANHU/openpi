#!/usr/bin/env bash
# 10-task 合并数据集训练(250 条 / 66463 帧,20k 步,每 2k 存 ckpt)。
# 分离式启动:setsid bash launch_train_10task.sh > /tmp/train_10task.log 2>&1 < /dev/null &
set -u
cd /data1/Franka_RealRobot/openpi || exit 1

export HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home
export OPENPI_DATA_HOME=/data1/vla-reasoning/openpi-cache
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
# 0.85 而不是 0.9:GPU0/GPU7 上有别人的进程(AI2-THOR 渲染 + serve.py),留 ~5.8G 余量
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export WANDB_MODE=offline
UVPY=$HOME/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
export PYTHONPATH=/data1/Franka_RealRobot/openpi/.venv/lib/python3.11/site-packages:\
/data1/Franka_RealRobot/openpi/src:/data1/Franka_RealRobot/openpi/packages/openpi-client/src

mkdir -p /data1/vla-reasoning/openpi-cache
exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$UVPY" scripts/train.py pi05_droid_franka_lora_10task \
  --exp-name=pi05_droid_franka_lora_10task_v0 --resume

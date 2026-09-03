#!/usr/bin/env bash
# 10task_v2: 同 10task 数据/超参, 数据 tasl_fr3_10task_v2 (物理删 dummy) + 尾 10 帧不做起点 (nonidle_ranges_tail10.json), 58760 起点, 30k 步, 每 5k 存 ckpt。
# 分离式启动:
#   setsid nohup bash /data1/Franka_RealRobot/data_pipeline/on_labserver/launch_train_10task_v2.sh > /tmp/train_10task_v2.log 2>&1 < /dev/null &
# 可覆盖: GPUS=0,5,6,7 MEMFRAC=0.85 EXP=pi05_droid_franka_lora_10task_v2_v0 bash launch_train_10task_v2.sh
# 起之前先 nvidia-smi: GPU 是共享的, 和别人共卡时 MEMFRAC 要降到 0.8 以下。
set -u
cd /data1/Franka_RealRobot/openpi || exit 1

export HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/data1/vla-reasoning/openpi-cache}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.85}
export WANDB_MODE=${WANDB_MODE:-online}   # 在线同步; 想离线: WANDB_MODE=offline bash ...
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
EXP=${EXP:-pi05_droid_franka_lora_10task_v2_v0}
PY=${PY:-/data1/Franka_RealRobot/openpi/.venv/bin/python}
export PYTHONPATH=/data1/Franka_RealRobot/openpi/.venv/lib/python3.11/site-packages:\
/data1/Franka_RealRobot/openpi/src:/data1/Franka_RealRobot/openpi/packages/openpi-client/src

[ -r /data1/Franka_RealRobot/filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json ] || { echo "缺 tail10 json"; exit 1; }
mkdir -p "$OPENPI_DATA_HOME" 2>/dev/null
echo "config=pi05_droid_franka_lora_10task_v2 exp=$EXP gpus=$GPUS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION"
exec env CUDA_VISIBLE_DEVICES=$GPUS "$PY" scripts/train.py pi05_droid_franka_lora_10task_v2 --exp-name="$EXP"

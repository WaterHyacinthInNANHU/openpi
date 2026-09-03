#!/usr/bin/env bash
# PBC LoRA fine-tune on the 10-task set: config pi05_pbc_franka_lora_10task_v2 (PBC base 199999 init,
# PBC norm stats, centre-crop geometry, dataset franka/tasl_fr3_10task_v2_pbc + tail-10 json, recipe = pi05_droid_franka_lora_10task_v2 (30k steps)).
# 分离式启动:
#   setsid nohup bash /data1/Franka_RealRobot/data_pipeline/on_labserver/pbc/launch_train_pbc_10task_v2.sh > /data1/Franka_RealRobot/openpi/logs/train_pbc_10task_v2.log 2>&1 < /dev/null &
# 可覆盖: GPUS=0,5,6,7 MEMFRAC=0.85 EXP=pi05_pbc_franka_lora_10task_v2_v0 bash launch_train_pbc_10task_v2.sh
# 起之前先 nvidia-smi: GPU 是共享的, 和别人共卡时 MEMFRAC 要降到 0.8 以下。
set -u
cd /data1/Franka_RealRobot/openpi || exit 1

export HF_LEROBOT_HOME=/data1/Franka_RealRobot/lerobot_home
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/data1/vla-reasoning/openpi-cache}
export JAXTYPING_DISABLE=1
export BEARTYPE_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.85}
export WANDB_MODE=${WANDB_MODE:-online}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
EXP=${EXP:-pi05_pbc_franka_lora_10task_v2_v0}
PY=${PY:-/data1/Franka_RealRobot/openpi/.venv/bin/python}
export PYTHONPATH=/data1/Franka_RealRobot/openpi/.venv/lib/python3.11/site-packages:\
/data1/Franka_RealRobot/openpi/src:/data1/Franka_RealRobot/openpi/packages/openpi-client/src

DS=/data1/Franka_RealRobot/lerobot_home/franka/tasl_fr3_10task_v2_pbc
PBC=/data1/Franka_RealRobot/checkpoints/axis_pi05_droid_plainbc_v1/199999
[ -r "$DS/meta/info.json" ] || { echo "缺 PBC 数据集 $DS (先跑 make_pbc_dataset.py)"; exit 1; }
grep -q '"pbc_geometry"' "$DS/meta/info.json" || { echo "$DS 不是 make_pbc_dataset.py 产出的 (info.json 无 pbc_geometry)"; exit 1; }
[ -r /data1/Franka_RealRobot/filters/tasl_fr3_10task_v2/nonidle_ranges_tail10.json ] || { echo "缺 tail10 json"; exit 1; }
[ -r "$PBC/params/_METADATA" ] || { echo "缺 PBC base params $PBC/params"; exit 1; }
[ -r "$PBC/assets/Devon018/Franka-Datasets-v2/norm_stats.json" ] || { echo "缺 PBC norm stats"; exit 1; }
mkdir -p "$OPENPI_DATA_HOME" 2>/dev/null
echo "config=pi05_pbc_franka_lora_10task_v2 exp=$EXP gpus=$GPUS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION"
exec env CUDA_VISIBLE_DEVICES=$GPUS "$PY" scripts/train.py pi05_pbc_franka_lora_10task_v2 --exp-name="$EXP"

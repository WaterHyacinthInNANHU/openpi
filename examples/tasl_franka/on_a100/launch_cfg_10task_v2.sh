#!/bin/bash
# AXIS Server 3 -- CFG cotrain 200k base + 10-task real-robot LoRA finetune (pi05_cotrain_franka_lora_10task_cfg_v2).
# Usage:  bash launch_cfg_10task_v2.sh                      # 8 GPUs, global batch 64, 16k steps
#         STEPS=50 EXP=cfg_dryrun bash launch_cfg_10task_v2.sh   # short dry run
#         BATCH=32 bash launch_cfg_10task_v2.sh
set -u

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
W=${W:-/localdisk/tasl_franka_finetune}      # checkpoints/, filters/ live here (shared with the PBC run)
CFG=pi05_cotrain_franka_lora_10task_cfg_v2
EXP=${EXP:-cotrain_cfg_10task_v2_b64}
BATCH=${BATCH:-64}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
MEMFRAC=${MEMFRAC:-0.92}
STEPS=${STEPS:-}                             # empty = the config's 16,000

source /localdisk/dihong_workspace/env.sh 2>/dev/null   # proxy for HF + caches on /localdisk
export PYTHONPATH=$REPO/src
export HF_LEROBOT_HOME=/localdisk/tasl_franka_cfg/lerobot_home   # our LeRobot v3.0 copy (lerobot 0.4.4 refuses v2.1)
export OPENPI_DATA_HOME=$W/cache/openpi
export CUDA_VISIBLE_DEVICES=$GPUS
export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC
export WANDB_MODE=offline
ulimit -n 1048576 2>/dev/null

# ---- preflight: refuse to start on a missing piece ----
FILTER=$W/filters/nonidle_ranges_tail10.json
BASE=/localdisk/dihong_workspace/runs/ckpts/pi05_axis_droid_cotrain_cfg/cotrain_cfg_phase/199999
DS=$HF_LEROBOT_HOME/ZhixuLi/tasl-fr3-10task-pbc-v2
[ -r "$FILTER" ] || { echo "FATAL: missing idle filter $FILTER"; exit 1; }
[ -d "$BASE/params" ] || { echo "FATAL: missing base params $BASE/params"; exit 1; }
[ -r "$BASE/assets/Devon018/Franka-Datasets-v2/norm_stats.json" ] || { echo "FATAL: missing base norm stats"; exit 1; }
N=$(ls "$DS/data/chunk-000/" 2>/dev/null | wc -l)
[ "$N" -eq 392 ] || { echo "FATAL: dataset incomplete, $N/392 parquet"; exit 1; }
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | wc -l)
[ "$BUSY" -eq 0 ] || { echo "FATAL: $BUSY compute process(es) already on the GPUs -- this box is shared, do not stack runs"; nvidia-smi; exit 1; }

echo "=========================================="
echo " config      : $CFG"
echo " exp         : $EXP"
echo " repo        : $REPO ($(git -C $REPO rev-parse --short HEAD), branch $(git -C $REPO rev-parse --abbrev-ref HEAD))"
echo " GPUs        : $GPUS  (n=$NGPU, fsdp=$NGPU, mem_frac=$MEMFRAC)"
echo " global batch: $BATCH   (per GPU $((BATCH/NGPU)))"
echo " steps       : ${STEPS:-16000 (config)}"
echo " base        : $BASE"
echo " norm stats  : base's own Devon018/Franka-Datasets-v2"
echo " data        : $DS  ($N parquet)"
echo " filter      : $FILTER"
echo " prompt tag  : Quality: 5, stage-2 dropout 0.15 / 0.05, seed 42"
echo "=========================================="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=========================================="

cd $REPO
exec $REPO/.venv/bin/python scripts/train.py $CFG \
    --exp-name=$EXP \
    --batch-size=$BATCH \
    --fsdp-devices=$NGPU \
    --num-workers=8 \
    ${STEPS:+--num-train-steps=$STEPS} \
    --overwrite
